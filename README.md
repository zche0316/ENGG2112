![The University of Sydney Logo](the-university-of-sydney-vector-logo.png)

# Project Report for ENGG2112

## Amazing Title of Project (which you should keep to within two lines)

---

### Authors

- Bob Smith, 510670000, Electrical Engineering
- Yuan Lin, 480421111, Biomedical Engineering
- David Mahendran, 49052222, Software Engineering

**Faculty of Engineering**

**Date: October 22, 2024**

---

## Executive Summary

Put your executive summary here. Its purpose is to summarize the objectives, methodology, main findings and conclusions of the project. Its intended audience is a reader who does not have enough time to read the main body of the report carefully, usually a busy executive, hence the name.

---

## Table of Contents

1. [Background and Motivation](#background-and-motivation)
2. [Objectives and Problem Statement](#objectives-and-problem-statement)
3. [Methodology](#methodology)
4. [Simulation Results](#simulation-results)
5. [Potential for Wider Adoption](#potential-for-wider-adoption)
6. [Conclusions](#conclusions)

---

## Background and Motivation

This project is focused on the use of machine learning to predict indigestion among university students, using a feature set consisting of (among others) gender, age, course enrolled in, blood pressure, and number of hours spent on study per semester week. It attempts to explain the curious and widely observed phenomenon of students reporting indigestion during the exam fortnight, particularly on their exam days. We have secured a promise of \$50,000 from the Deputy Vice Chancellor Education office to study it more deeply.

Our data set was obtained from **DataSets-R-Us.com**, a reputable open source data repository, collected from the University of South Antarctica through a poll of its 200 students across all academic departments, and had a total of 25 features or independent variables.

## Objectives and Problem Statement

The main objective is to use some or all of the available features to find the probability of a student having indigestion sometime during the examination period, as well as on the day of any scheduled examination. We denote the features as \( x_i \), \( i = 1,\ldots,25 \), and the target variables \( y \) and \( z \) as indicator functions of whether a student has indigestion sometime in the exam period or on the day of their scheduled exams.

By comparing \( f_y \) and \( f_z \) to a user-defined threshold, we will then be able to make binary decisions on whether indigestion will occur.

## Methodology

### Data Pre-Processing

- **One-Hot Encoding**: Convert categorical variables into binary numerical variables using `onehotencode()`.
- **Data Cleaning**: Replace missing values with zero or the average using `dataclean()` and `average()`.

### Feature Extraction

Used `feature1()` and `feature2()` from **sklearn** to correlate features with target variables using statistical correlations.

### Classification

Tried the Thing One and Thing Two methods. Thing Three was considered but discarded due to complexity. Simulations ran on an 80-20 training-testing split in Google Colab, using accuracy and AUC as measures.

### Simulation Environment

Random cross-validation of ten instances, with the final model chosen through averaging the three most accurate models.

## Simulation Results

### Key Findings and Significance

Our method proved highly competitive against existing methods in literature, such as in [Einstein's study](#).

![Performance Curve](Dec-Tree.png)

### Issues Faced

- Problem One
- Problem Two

## Potential for Wider Adoption

Future improvements include increasing data variety and using sophisticated methods. Strong industry interest, as evidenced by a Deloitte market study. No known commercial solutions exist currently.

## Conclusions

The project was somewhat successful. We achieved up to 85% accuracy. Future work could push this to 95%, enabling potential commercialization as outlined above.

---

## References

- [Einstein Study](#)
- [Deloitte Market Study](#)

