---
layout: page
title: "papers"
date: 2014-02-11 00:07
comments: true
sharing: true
footer: true
---
## Code Compaction of an Operating System Kernel

Download: [pdf](/assets/pages/papers/code_compaction_kernel.pdf) <br>
Authors: Haifeng He, John Trimble, Somu Perianayagam, Saumya Debray, Gregory Andrews

### Abstract

General-purpose operating systems, such as Linux, are increasingly being used in embedded systems. Computational resources are usually limited, and embedded processors often have a limited amount of memory. This makes code size especially important. This paper describes techniques for automatically reducing the memory footprint of general-purpose operating systems on embedded platforms. The problem is complicated by the fact that kernel code tends to be quite different from ordinary application code, including the presence of significant amount of hand-written assembly code, multiple entry points, implicit control flow paths involving interrupt handlers, and a significant amount of indirect control flow via function pointers. We use a novel “approximate decompilation” technique to apply source-level program analysis to hand-written assembly code. A prototype implementation of our ideas, applied to a Linux kernel that has been configured to exclude unneccessary code, obtains a code size reduction of over 25%.

## Combining High Level Alias Analysis with Low Level Code Compaction of the Linux Kernel

Download: [pdf](/assets/pages/papers/honors_thesis.pdf) <br>
Authors: John Trimble

### Abstract

The limited resources of embedded devices make it both costly and difficult to deploy with general-purpose operating systems such as Linux. The use of low level code compaction techniques can reduce the code size of the kernel to better suit the environment of an embedded device. However, using low level code compaction has typically precluded the use of high level aliasing information. This makes it difficult to resolve potential targets of indirect calls and which in turn reduces the degree of code compaction. This honors thesis discusses a method for combining high level aliasing information with low level code compaction of the Linux kernel by having the high level analysis construct part of the kernel call graph.