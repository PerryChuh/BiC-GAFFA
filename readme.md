# Overcoming Lower-Level Constraints in Bilevel Optimization: A Novel Approach with Regularized Gap Functions

This project is a supporting codebase for the paper titled *Overcoming Lower-Level Constraints in Bilevel Optimization: A Novel Approach with Regularized Gap Functions*. 

It implements the methods and experiments described in the paper, focusing primarily on the content in Section 6 and Appendix A.2.

## Structure

The project directory is organized as follows:

```bash
BiC-GAFFA-main/
│
├── 1. Toy/ # Files for the synthetic problem
│ ├── Toy.py  # main file for sensitivity analysis
| ├── Toy_E-AiPOD.py
| ├── Toy_LVHBA.py
│ └── Toy_utils.py
├── 2.1 HO_Sparse Group LASSO/ # Files for SGL problem
│ ├── Data_Generator.py
│ ├── gLasso_data.py # Generate data and stored
│ ├── gLasso_GAFFA.py # BiC-GAFFA for SGL problem 
│ ├── gLasso_main.py # main file for this experiments
│ ├── HC_SGL.py
│ ├── SGL_Algorithms.py
│ └── utils.py
├── 2.2 HO_Support Vector Mahchine/  # Files for SVM
│ ├── HO_SVM_GAFFA.py # BiC-GAFFA for SVM
│ ├── HO_SVM_GAM.py
│ └── HO_SVM_LV-HBA.py
├── 2.3 HO_Data HyperClean/ # Files for HyperClean
│ ├── Hyperclean_GAFFA.py   # BiC-GAFFA for HyperClean
│ ├── Hyperclean_GAM.py
│ └── Hyperclean_LVHBA.py
├── 3. GAN/ # Files for different type of GANs
│ ├── Bi_*gan.py  #  BiC-GAFFA based GANsw
│ ├── *gan.py     #  other gan experiments files
│ ├── gan_main.py   #  main file
│ ├── mixture_gaussian.py   #  The distributions
│ ├── models.py   # Generators and Discriminators
│ └── utils.py    # Earth mover's distance function  
└── README.md # Project README
```

## Usage 

To reproduce the data used in our paper, simply run the corresponding files in their respective folders.

1. **Synthetic Problem Sensitivity Analysis (Section 6.1):**
   - Run `Toy.py` in the `1. Toy` folder.
   - `main0()` corresponds to Figure 1.
   - `main1()` corresponds to Tables 1, 4, and 5.
   - `main2()` corresponds to Figure 2.
2. **Sparse Group Lasso Problem (Section 6.2):**
   - First, run `gLasso_data.py`.
   - Then, run `gLasso_main.py`.
3. **SVM/Hyperclean Problem (Section 6.2):**
   - Run each file in the relevant folder sequentially.
   - Perform any necessary post-processing tasks afterward.
4. **GAN Results (Section 6.3):**
   - Run `gan_main.py` in the `3. GAN` folder.
   - You may skip any methods you are not interested in.

All results are stored in `.csv` format, which can be easily loaded using `pandas`.



