use std::ops::Range;

use criterion::{
    BatchSize, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use rand::prelude::*;
use rand::rngs::StdRng;
use text::{Anchor, Bias, Buffer, BufferId, Point, ReplicaId, TextSummary};
use util::RandomCharIter;

/// Sizes (in bytes) of the buffers the benchmarks operate on. They roughly
/// correspond to a small source file and a large one.
const SIZES: [usize; 2] = [4096, 65536];

/// Returns a random string whose UTF-8 length is close to but no more than
/// `len` bytes, biased towards characters that occur in source code.
fn generate_random_text(rng: &mut StdRng, len: usize) -> String {
    let mut text = String::with_capacity(len);
    let mut chars = RandomCharIter::new(rng);
    loop {
        let ch = chars.next().unwrap();
        if text.len() + ch.len_utf8() > len {
            break;
        }
        text.push(ch);
    }
    text
}

fn build_buffer(rng: &mut StdRng, len: usize) -> Buffer {
    let text = generate_random_text(rng, len);
    Buffer::new(ReplicaId::LOCAL, BufferId::new(1).unwrap(), text)
}

/// Returns `count` offsets that all land on character boundaries.
fn random_offsets(rng: &mut StdRng, buffer: &Buffer, count: usize) -> Vec<usize> {
    let snapshot = buffer.snapshot();
    (0..count)
        .map(|_| snapshot.clip_offset(rng.random_range(0..=snapshot.len()), Bias::Left))
        .collect()
}

fn random_ranges(rng: &mut StdRng, buffer: &Buffer, count: usize) -> Vec<Range<usize>> {
    let max_len = 128;
    let snapshot = buffer.snapshot();
    (0..count)
        .map(|_| {
            let start = snapshot.clip_offset(rng.random_range(0..=snapshot.len()), Bias::Left);
            let end = snapshot.clip_offset(
                (start + rng.random_range(0..max_len)).min(snapshot.len()),
                Bias::Right,
            );
            start..end
        })
        .collect()
}

fn new_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Buffer::new");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let text = generate_random_text(&mut rng, size);

        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &text, |b, text| {
            b.iter(|| {
                black_box(Buffer::new(
                    ReplicaId::LOCAL,
                    BufferId::new(1).unwrap(),
                    text.clone(),
                ));
            });
        });
    }
    group.finish();
}

fn edit_benchmark(c: &mut Criterion) {
    let edits_per_iteration = 32;

    let mut group = c.benchmark_group("Buffer::edit");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let buffer = build_buffer(&mut rng, size);
        let edits = random_ranges(&mut rng, &buffer, edits_per_iteration)
            .into_iter()
            .map(|range| (range, generate_random_text(&mut rng, 16)))
            .collect::<Vec<_>>();

        group.throughput(Throughput::Elements(edits_per_iteration as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &edits, |b, edits| {
            b.iter_batched_ref(
                || buffer.branch(),
                |buffer| {
                    for (range, text) in edits {
                        black_box(buffer.edit([(range.clone(), text.as_str())]));
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn anchor_benchmark(c: &mut Criterion) {
    let anchor_count = 1024;

    let mut group = c.benchmark_group("Buffer::summaries_for_anchors");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let mut buffer = build_buffer(&mut rng, size);

        // `summaries_for_anchors` requires the anchors to be sorted, which is how
        // callers (selections, diagnostics, ...) store them.
        let mut offsets = random_offsets(&mut rng, &buffer, anchor_count);
        offsets.sort_unstable();
        let anchors = offsets
            .into_iter()
            .map(|offset| buffer.snapshot().anchor_at(offset, Bias::Left))
            .collect::<Vec<Anchor>>();

        // Edit the buffer after taking the anchors so that resolving them has to
        // account for the intervening operations, like it does while editing.
        for (range, text) in random_ranges(&mut rng, &buffer, 32)
            .into_iter()
            .map(|range| (range, generate_random_text(&mut rng, 16)))
        {
            buffer.edit([(range, text.as_str())]);
        }

        let snapshot = buffer.snapshot().clone();
        group.throughput(Throughput::Elements(anchor_count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &anchors, |b, anchors| {
            b.iter(|| {
                for summary in snapshot.summaries_for_anchors::<usize, _>(anchors.iter().copied()) {
                    black_box(summary);
                }
            });
        });
    }
    group.finish();
}

fn edits_since_benchmark(c: &mut Criterion) {
    let edit_count = 128;

    let mut group = c.benchmark_group("Buffer::edits_since");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let mut buffer = build_buffer(&mut rng, size);
        let version = buffer.version();

        for (range, text) in random_ranges(&mut rng, &buffer, edit_count)
            .into_iter()
            .map(|range| (range, generate_random_text(&mut rng, 16)))
        {
            buffer.edit([(range, text.as_str())]);
        }

        let snapshot = buffer.snapshot().clone();
        group.throughput(Throughput::Elements(edit_count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &version, |b, version| {
            b.iter(|| {
                for edit in snapshot.edits_since::<Point>(version) {
                    black_box(edit);
                }
            });
        });
    }
    group.finish();
}

fn text_summary_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Buffer::text_summary_for_range");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let buffer = build_buffer(&mut rng, size);
        let ranges = random_ranges(&mut rng, &buffer, 256);
        let snapshot = buffer.snapshot().clone();

        group.throughput(Throughput::Elements(ranges.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &ranges, |b, ranges| {
            b.iter(|| {
                for range in ranges {
                    black_box(snapshot.text_summary_for_range::<TextSummary, _>(range.clone()));
                }
            });
        });
    }
    group.finish();
}

fn point_to_offset_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Buffer::point_to_offset");
    for size in SIZES {
        let mut rng = StdRng::seed_from_u64(1);
        let buffer = build_buffer(&mut rng, size);
        let snapshot = buffer.snapshot().clone();
        let points = random_offsets(&mut rng, &buffer, 256)
            .into_iter()
            .map(|offset| snapshot.offset_to_point(offset))
            .collect::<Vec<Point>>();

        group.throughput(Throughput::Elements(points.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &points, |b, points| {
            b.iter(|| {
                for point in points {
                    black_box(snapshot.point_to_offset(*point));
                }
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    new_benchmark,
    edit_benchmark,
    anchor_benchmark,
    edits_since_benchmark,
    text_summary_benchmark,
    point_to_offset_benchmark,
);
criterion_main!(benches);
