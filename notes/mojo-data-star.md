+++
title = "Building mojo-data-star: Interactive Mandelbrot Explorer with Mojo 🔥"
description = "An evolving project log building a GPU-accelerated Mandelbrot explorer with Mojo, FastAPI, and Datastar."
date = 2026-01-21
draft = true
[taxonomies]
tags = ["mojo", "gpu", "python", "fastapi", "datastar"]
+++

Mojo has become my favourite way to explore the line between "nice to write" and "fast enough to feel like magic". In **mojo-data-star**, I'm applying that to an old classic: the Mandelbrot set.

Rather than another static image renderer, the goal here is an **interactive Mandelbrot explorer** that runs a Mojo kernel on the GPU, and streams the result into a modern web front end using **FastAPI**, **FastHTML**, and **Datastar**.

## Project goals

At a high level, I want this project to:

- Demonstrate a clean pattern for **Mojo ⇄ Python interop** using a compiled extension module.
- Provide a realistic example of driving **GPU kernels from a web app**.
- Show how to structure tests across **Mojo's TestSuite** and **pytest**.
- Grow over time into a reference for future DataBooth experiments.

This post will evolve as the project does, recording the steps, trade-offs, and occasional missteps along the way.

## Step 1 – Establish a CPU baseline in Mojo

The original sketch for this project used earlier Mojo syntax, including `let` and `alias`, and relied on intrinsics that have since shifted. The first job was simply to **modernise the core Mandelbrot computation** for Mojo 0.25.x.

The current implementation lives in `mandelbrot.mojo` and exposes a single Python-facing function:

- `compute_mandelbrot(width, height, xmin, ymin, xmax, ymax) -> Tensor[Int32]`

Internally, it:

- Uses `comptime let` for constants (for example, `MAX_ITERS`).
- Uses `var` for mutable values inside the iteration loop.
- Fills a `Tensor[Int32]` with escape-time iteration counts for each pixel.

This is still a **CPU implementation**, but the structure mirrors what we will want on the GPU: a tight inner loop over complex coordinates, with a simple interface for the host to call.

## Step 2 – Wire up a minimal FastAPI front end

Before leaning into GPUs, I wanted a quick way to "see" the fractal. A small **FastAPI** app does the job for now:

- `/` serves a bare-bones HTML page with an `<img>` tag.
- `/render` accepts view parameters (`xmin`, `ymin`, `xmax`, `ymax`), calls `compute_mandelbrot`, and converts the iteration counts into a PNG using Pillow.

This mirrors the behaviour of the original FastHTML + Datastar concept sketch, but keeps the plumbing intentionally simple so the focus stays on the Mojo side.

In later iterations I plan to:

- Replace the HTML with a proper **FastHTML + Datastar** front end.
- Introduce richer controls (zoom, pan, max-iteration sliders, presets).
- Experiment with progressive refinement techniques for smoother interaction.

## Step 3 – Testing from the start

Even for an exploratory project, I want a solid testing story. The current plan is:

- Use Mojo's **TestSuite** to exercise the `compute_mandelbrot` implementation directly.
- Use **pytest** to test the Python integration layer (colour mapping, HTTP endpoints, and error handling).

Starting with tests forces me to keep the public surface area small and predictable, which should pay off once the GPU kernel arrives.

## Next steps

From here, the roadmap looks roughly like this:

1. Swap the CPU loop for a **GPU kernel** using Mojo's current GPU APIs.
2. Tune grid and block sizes for a few common GPUs.
3. Flesh out the FastHTML + Datastar UI to make the explorer genuinely fun to use.
4. Capture some benchmarks and screenshots for a follow-up post.

As mojo-data-star evolves, I will update this draft with new sections, code snippets, and performance notes.