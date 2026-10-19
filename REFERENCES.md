# References

## How this list was built

Part A lists standard textbooks, foundational algorithm papers and software
papers that are widely cited and whose bibliographic details are well
established. Before submission, **check every entry against the publisher's
page or DOI**. This document was prepared without live database access, so
page numbers and DOIs should be confirmed by the student.

Part B lists literature that is relevant context but whose details were **not
verified** while writing. Do not cite anything from Part B until you have found
and read the paper yourself.

No reference here is used to support a numerical claim about this project's
results. All results come from this software running on synthetic data.

---

## Part A - Established references (confirm DOI/pages before citing)

### Antenna and microwave engineering
1. C. A. Balanis, *Antenna Theory: Analysis and Design*, 4th ed. Hoboken, NJ: Wiley, 2016.
   (Patch-antenna resonance, effective permittivity, fringing length extension, gain/directivity.)
2. D. M. Pozar, *Microwave Engineering*, 4th ed. Hoboken, NJ: Wiley, 2012.
   (Reflection coefficient, S-parameters, return loss, mismatch loss, resonator Q.)
3. C. L. Holloway, E. F. Kuester, J. A. Gordon, J. O'Hara, J. Booth, D. R. Smith,
   "An overview of the theory and applications of metasurfaces: The two-dimensional
   equivalents of metamaterials," *IEEE Antennas and Propagation Magazine*, vol. 54,
   no. 2, pp. 10-35, 2012. doi:10.1109/MAP.2012.6230714

### Datasets used
* E. S. Saçın and A. C. Durgun, "Microstrip Patch Antenna Modeling Using Neural Networks with
  Knowledge-Based Regularization," *Neural Computing and Applications*, vol. 37, pp. 3827–3837, 2025.
  doi:10.1007/s00521-024-10860-5. Dataset: https://github.com/METU-SIMPL/Microstrip-Patch (commit
  7113818a). Details taken from the dataset's own README (citation requested by the authors).
* Zenodo record 15866865, https://zenodo.org/records/15866865: **NOT inspected** (unreachable from
  the build environment); its title, authors and license are unknown. Do not cite it until you have
  opened the record.

### Machine-learning algorithms
4. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001.
   doi:10.1023/A:1010933404324 (Random Forest and permutation importance.)
5. P. Geurts, D. Ernst, L. Wehenkel, "Extremely randomized trees," *Machine Learning*,
   vol. 63, no. 1, pp. 3-42, 2006. doi:10.1007/s10994-006-6226-1
6. J. H. Friedman, "Greedy function approximation: A gradient boosting machine,"
   *Annals of Statistics*, vol. 29, no. 5, pp. 1189-1232, 2001. doi:10.1214/aos/1013203451
   (Gradient boosting and partial dependence plots.)
7. C. E. Rasmussen, C. K. I. Williams, *Gaussian Processes for Machine Learning*.
   Cambridge, MA: MIT Press, 2006.
8. A. Fisher, C. Rudin, F. Dominici, "All models are wrong, but many are useful:
   Learning a variable's importance by studying an entire class of prediction models
   simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1-81, 2019.
9. N. Meinshausen, "Quantile regression forests," *Journal of Machine Learning Research*,
   vol. 7, pp. 983-999, 2006. (Context for forest-based uncertainty; not implemented here.)
10. V. Kuleshov, N. Fenner, S. Ermon, "Accurate uncertainties for deep learning using
    calibrated regression," in *Proc. 35th ICML*, PMLR 80, 2018. (Context: why
    uncalibrated uncertainty must not be presented as confidence intervals.)
11. C. Molnar, *Interpretable Machine Learning*, online book, https://christophm.github.io/interpretable-ml-book/
    (Overview of permutation importance and PDP and their pitfalls.)

### Optimization and design of experiments
12. R. Storn, K. Price, "Differential evolution - A simple and efficient heuristic for
    global optimization over continuous spaces," *Journal of Global Optimization*,
    vol. 11, no. 4, pp. 341-359, 1997. doi:10.1023/A:1008202821328
13. J. A. Nelder, R. Mead, "A simplex method for function minimization," *The Computer
    Journal*, vol. 7, no. 4, pp. 308-313, 1965. doi:10.1093/comjnl/7.4.308
14. C. Tsallis, D. A. Stariolo, "Generalized simulated annealing," *Physica A*, vol. 233,
    pp. 395-406, 1996. doi:10.1016/S0378-4371(96)00271-3
15. Y. Xiang, S. Gubian, B. Suomela, J. Hoeng, "Generalized simulated annealing for global
    optimization: The GenSA package," *The R Journal*, vol. 5, no. 1, pp. 13-28, 2013.
    (Algorithm basis cited by SciPy's `dual_annealing`.)
16. M. D. McKay, R. J. Beckman, W. J. Conover, "A comparison of three methods for selecting
    values of input variables in the analysis of output from a computer code,"
    *Technometrics*, vol. 21, no. 2, pp. 239-245, 1979. (Latin hypercube sampling.)
17. D. R. Jones, M. Schonlau, W. J. Welch, "Efficient global optimization of expensive
    black-box functions," *Journal of Global Optimization*, vol. 13, pp. 455-492, 1998.
    (Context: surrogate/GP-based optimization of expensive simulators.)
18. K. Deb, A. Pratap, S. Agarwal, T. Meyarivan, "A fast and elitist multiobjective genetic
    algorithm: NSGA-II," *IEEE Transactions on Evolutionary Computation*, vol. 6, no. 2,
    pp. 182-197, 2002. (Context for multi-objective optimization; not implemented.)
19. S. Koziel, S. Ogurtsov, *Antenna Design by Simulation-Driven Optimization*.
    Cham: Springer (SpringerBriefs in Optimization), 2014.

### Software
20. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine
    Learning Research*, vol. 12, pp. 2825-2830, 2011.
21. P. Virtanen et al., "SciPy 1.0: Fundamental algorithms for scientific computing in
    Python," *Nature Methods*, vol. 17, pp. 261-272, 2020. doi:10.1038/s41592-019-0686-2
22. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362,
    2020. doi:10.1038/s41586-020-2649-2
23. Software documentation (cite as web resources with access date): Streamlit
    (https://docs.streamlit.io), SQLAlchemy (https://docs.sqlalchemy.org), SQLite
    (https://sqlite.org), ReportLab (https://docs.reportlab.com), Plotly (https://plotly.com/python/).

---

## Part B - UNVERIFIED context (verify before citing)

These items are relevant to ML-assisted antenna design. Their exact details were
**not** checked while writing. Find each one in IEEE Xplore / Wiley / Google
Scholar, read it, and correct the details before using it.

* [UNVERIFIED] H. M. El Misilmani, T. Naous, S. K. Al Khatib, "A review on the design and
  optimization of antennas using machine learning algorithms and techniques,"
  *International Journal of RF and Microwave Computer-Aided Engineering*, 2020.
* [UNVERIFIED] N. Sarker et al., review of machine learning and deep learning in antenna
  design, optimization and selection, *IEEE Access*, 2023.
* [UNVERIFIED] Search terms for building your own literature review: "machine learning
  assisted antenna design", "surrogate model metasurface antenna", "Gaussian process
  antenna optimization", "inverse design metasurface deep learning".

## Statement on the literature review

`PROJECT_REPORT.md` §5 describes the research area in general terms only. It does
not attribute specific numerical results to any paper. Before submission, add 5-10
papers you have read yourself, with their verified details.
