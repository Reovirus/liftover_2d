# coolliftover
- [Report](https://docs.google.com/document/d/1z8f_V4XeqGB1uhX68kY3jnS_5l7sKFrqBRg1bUkJyCc/edit?usp=sharing)
- [About project](https://github.com/Reovirus/liftover_2d/blob/main/about_project.md)
# GSOC 2025 Final Outcome Report

As part of the Google Summer of Code cohort of 2025, with NumFOCUS (Open2C mentorship) we have implemented LiftOver2D, a novel library, that facilitates fast conversion of chromosome contact data (Hi-C) between genome assemblies, as well as a related subtask of increasing the resolution of Hi-C using only the count matrix.

During the project, the following was implemented: [https://github.com/Reovirus/liftover_2d/commits/main/](https://github.com/Reovirus/liftover_2d/commits/main/)

- [Tools for reading and working with HiC, pairs, and chain formats](https://github.com/Reovirus/liftover_2d/tree/main/src/readers)
- [Integration with the polars-bio library](https://github.com/Reovirus/liftover_2d/blob/main/src/transformers/bin_remapper.py)
- [Three models for Hi-C pixel partitioning (CVD Normalisation, linear approximation, polynomial interpolation)](https://github.com/Reovirus/liftover_2d/tree/main/src/pixel_dividers)
- [Python interfaces for model calls](https://github.com/Reovirus/liftover_2d/blob/main/test.ipynb)
- An object-oriented code structure, with a focus on generalization and minimizing overhead when adding new methods, making the project easier to scale-up and maintain  

To evaluate the quality of the transformations, we developed a new metric: we compare the transformed matrix with experimental data from which a part of the contacts was deliberately excluded. This allows us to estimate how much information is lost during conversion. We can use any type of difference like distance (MDE, wasserstein, spectral…)

During the work, I faced several challenges:

- The polars-bio documentation was incomplete, so I had to rely on practical case studies  
- Some libraries lacked necessary functionality (but we’ve found solutions)  
- Memory limitations (RAM) became a bottleneck when computing distances between matrices  

In addition, I often had to balance code quality with speed of development and be more attentive to deadlines and reporting — these became the main personal constraints during the project (and the biggest blocker).

At this stage, the basic models and the comparison algorithm have been implemented (although there are still long computations of Wasserstein distances ahead). To bring the project to a fully functional tool, the following work remains:

- Testing and benchmarking of models  
- Building visualizations  
- Performance optimizations (for example, implementing polynomial interpolation in a lower-level language)  
- Defining requirements for accuracy and speed  
- Realisation in our CI  

I plan to continue working on LiftOver2D at a calmer pace, with less strict deadlines, and — if the organization is interested — bring it to a production-ready state.

I would like to thank my mentors and the organization for their support, advice, and the opportunity to participate in GSoC. This experience allowed me to once again immerse myself in the world of DNA-DNA contacts and significantly improve my skills in object-oriented programming, adapting methods into unified interfaces, and designing metrics and benchmarks (and hopefully soon — in their full implementation).

