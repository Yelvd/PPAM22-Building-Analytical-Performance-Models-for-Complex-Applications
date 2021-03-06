# PPAM22 Building Analytical Performance Models-for-Complex-Applications

This repositry contains the data, figures, and scripts used for our paper __"Building Fine-Grained Analytical Performance Models for Complex Applications"__

## Structure
- **experiment-template**: Template for running experiments
- **figs**: Containes all figures.
  - **figs/** Figures generated by scripts in repo
  - **figs/external** Figures from external sources
- **scripts**: Contains the scripts used to parse the data, and create the figures
- **results**: Contains the raw data for all experiments. 
  - *.csv datafiles that store extracted from the *.cubex files
  - *.cubex: output files from SCORE-P, contains timings per function per process
  - RBC.*.xmf / Fluid*.xml: Internal Hemocell output files, stores the simulation data. Can be read in using Paravieuw.
- **Tables.pdf**: A PDF with the tables containing all the raw data for the generated plots.
