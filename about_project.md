# Final Work Report – GSoC 2025 at Open2C (LiftOver2D)

**Contributor:** Egor Pitikov  
**Mentors:** phlya, VOY, Aleksandra Galitsyna  
**Organisation:** NumFocus  

## LiftOver2D: A Tool for Converting Hi-C Matrices Across Assemblies and Binning Schemes

### About the Project
LiftOver2D is a project designed to quickly translate chromosomal contact data (Hi-C) between different genome assemblies. Additionally, it allows increasing the resolution of Hi-C data using only the counts matrix. The main goal is to provide user-friendly and extensible Python tools for working with Hi-C data, making it easy to add new methods and models. This project was carried out as part of Google Summer of Code 2025.

### Motivation
Currently, there is no easy way to quickly and accurately convert Hi-C data between assemblies. Existing solutions either discard incomplete pixel mappings or require reprocessing from raw reads, which is slow and resource-intensive. LiftOver2D addresses this by enabling direct pixel-level conversion without using raw reads.

### My Contribution
During the project, I:  
* Implemented tools to read and work with Hi-C, pairs, and chain formats.  
* Integrated the project with the polars-bio library.  
* Developed three models for Hi-C pixel splitting: CVD Normalisation, linear approximation, and polynomial interpolation.  
* Created Python interfaces for calling these models easily.  
* Built an object-oriented code structure focused on scalability and easy addition of new features.

### Architecture and Implementation
The project was designed using object-oriented principles: methods follow a unified interface, which simplifies the integration of new models. This approach reduces maintenance effort and makes the code more extensible.

To evaluate conversion quality, a metric was developed that compares the transformed matrix with experimental data from which some contacts were intentionally removed. Several types of metrics are supported, including MDE, Wasserstein, and spectral distances.

### Results
At this stage, the basic models and comparison algorithm have been implemented (although Wasserstein distance computations are still time-consuming). The project demonstrates the possibility of building a scalable and extensible architecture for working with Hi-C matrices, as well as using metrics for quantitative evaluation of conversions.

### Future Work
To turn LiftOver2D into a fully functional tool, the next steps include:  
* Testing and benchmarking the models.  
* Building visualizations.  
* Optimizing performance (e.g., moving polynomial interpolation to a lower-level language).  
* Defining accuracy and speed requirements.  
* Integrating the project into CI.

I plan to continue working on LiftOver2D at a more relaxed pace, without strict deadlines, and — if the organization is interested — bring it to a state ready for full use.

### Acknowledgements
I would like to thank my mentors for their valuable feedback and constructive criticism — they helped me get involved in the community and start writing modern, clear code. I also thank the Open2C community for their ideas and feedback, and NumFocus for making this project possible.
