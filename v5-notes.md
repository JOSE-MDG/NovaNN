# NovaNN v5

> **NovaNN v5.0.0 is under active development.** This note explains what v5 will be compared to the current stable version (v4.0.4). For the current stable docs, see [README.md](./README.md).

## Table of Contents

- [NovaNN v5](#novann-v5)
  - [Table of Contents](#table-of-contents)
  - [What is NovaNN v5?](#what-is-novann-v5)
  - [Project philosophy](#project-philosophy)
  - [A new core instead of NumPy](#a-new-core-instead-of-numpy)
  - [More than a framework](#more-than-a-framework)
  - [The same Python API](#the-same-python-api)
  - [Built with help from AI agents](#built-with-help-from-ai-agents)
  - [Roadmap and current status](#roadmap-and-current-status)

## What is NovaNN v5?

v5 is NovaNN rewritten from scratch. The v4 code is not getting upgraded little by little, it is getting replaced. A new native core is being built next to it, and when that core is ready it takes over completely. Until then, v4.0.4 stays as the stable version on PyPI, and the Python package in this repo is still v4 code.

NovaNN v4 is a deep learning framework written in Python on top of NumPy. It only runs on CPU and covers the usual pieces (tensors, autograd, layers, optimizers, metrics). It was built to learn how frameworks like PyTorch work on the inside, not to compete with them.

NovaNN v5 keeps that same Python API and that same spirit, but everything underneath changes, and the project goes further. It stops being just a framework for training small models and becomes one tool for the whole life of a model, running on real hardware.

## Project philosophy

v5 comes from a different frustration. Training a model today means gluing together several libraries: one for tensors, one for transformers, one for datasets, one for quantization, one for adapters, one for trainers. Each with its own way of doing things. The idea behind v5 is to have all of that in one place, so the whole process from pretraining to deployment lives in a single tool.

## A new core instead of NumPy

The biggest technical change is that NumPy goes away. In v4 every operation is a NumPy call on CPU, which is great for reading and learning but slow, wasteful with memory, and stuck on one device. In v5 the computation runs natively on CPU and GPU, with memory managed outside of Python.

As a user you will notice this in two ways. First, you will be able to choose where things run: CPU, CUDA, or HIP on Linux. Second, you will get the precisions that modern models need: FP16, BF16, FP8, and FP4 next to the classic types, which opens the door to mixed precision training and quantization. Both show up as simple new arguments on the tensors and modules you already know, so there is nothing new to learn.

The practical result is speed without rewriting your code. And since the core is no longer pure Python, installing v5 means compiling it for your hardware instead of a plain `pip install`, using the CMake presets documented for contributors.

## More than a framework

The biggest conceptual change is scope. v4 ends when a classifier finishes training. v5 treats training as one step among many, and wants pretraining, fine-tuning, quantization, adapters, inference, and deployment to live in the same tool:

- Transformer architectures and dataset handling, which v4 leaves to other projects.
- Quantization workflows (NF4, FP8, and similar formats) for serving large models.
- Efficient fine-tuning with adapters such as LoRA and QLoRA.
- SFT and DPO trainers like the ones in dedicated training libraries.
- Inference, deployment, and later on a model hub.

None of this lands all at once. It arrives piece by piece, but the direction is set: everything v4 covers becomes the base of v5, not its limit.

## The same Python API

Even with a new core and a wider scope, v5 keeps the API from v4. Tensors, `nn.Module`, `functional`, `optim`, `metrics`, `save` and `load` all stay, and the usual loop of forward, backward, and optimizer step works the same way. What changes is what happens underneath: those calls go to native code instead of NumPy, and native errors come back as normal Python exceptions.

New things in the API only appear where v5 does something genuinely new: devices, low precision types, quantization, adapters, trainers, and hub access. The rest stays where it is, so what you know from v4 still applies.

## Built with help from AI agents

v4 was written fully by hand. v5 is developed with help from AI agents, so the repo includes the tooling for that: reusable skills for the languages and workflows used in the native core, feature specs in `specs/` written before implementing, and generated files that get regenerated from templates instead of edited by hand. Humans and agents follow the same rules, documented in `AGENTS.md`.

## Roadmap and current status

v4.0.4 stays as the stable version on PyPI. From here the work goes through a native core alpha (runtimes, backends, types, memory), then a beta with bindings and autograd (the native Python API and the math operations), and finally the full v5.0.0 with transformers, quantization, adapters, trainers, and hub support.

Right now the Python package in `nova/` is still v4.0.4 and the native core is being built backend by backend, so read this file as the plan, not as a description of what is already there.
