# Horizon-Benchmark

This repository provides the code to construct the `HORIZON benchmark` — a large-scale, cross-domain benchmark built by refactoring the popular **Amazon-Reviews 2023 datase**t for evaluating sequential recommendation and user behavior modeling.
We do not release any new data; instead, we share reproducible scripts and guidelines to regenerate the benchmark, enabling rigorous evaluation of generalization across time, unseen users, and long user histories. The benchmark supports modern research needs by focusing on temporal robustness, out-of-distribution generalization, and long-horizon user modeling beyond next-item prediction.

## Overview 

HORIZON is a benchmark for in-the-wild user modeling in the e-commerce domain. This repository provides the necessary code to load a publicly available dataset, process it to create a benchmark, and then run a diverse set of user modeling algorithms on the benchmark. The publicly available dataset was collected from amazon.com, likely representing users from the United States.   

## Objective 

Our objective is to provide a standardized testbed for user modeling.  

## Audience 

HORIZON benchmark is intended for researchers, AI practitioners, and industry professionals who are interested in evaluating user modeling algorithms. 

## Intended Uses 

HORIZON benchmark can be used as a standardized evaluation platform to evaluate performance of both existing and new algorithms. Our results may be most useful for settings involving products in similar categories to the dataset we used. For a list of these 33 categories, see https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/blob/main/all_categories.txt. 

## Out of Scope Uses 

HORIZON benchmark is not intended to be used to circumvent any policies adopted by LLM providers. 

The user modeling algorithms provided in HORIZON are for e-commerce product scenarios only and may not translate to other kinds of products or buying behavior. 

## Evaluation 

We have evaluated many state-of-the-art algorithms on the HORIZON benchmark. For details, please refer to the accompanying [Arxiv paper](TBD). 

## Limitations 

- HORIZON provides an offline evaluation. In real-world applications, offline evaluation results may differ from online evaluation that involves deploying a user modeling algorithm. 

- HORIZON benchmark contains only items in English language. 

- The accuracy of HORIZON evaluation metrics for a real-world application depends on the diversity and representativeness of the underlying data.

## Usage 

This project is primarily designed for research and experimental purposes. We strongly recommend conducting further testing and validation before considering its application in industrial or real-world scenarios. 

## Feedback and Collaboration 

We welcome feedback and collaboration from our audience. If you have suggestions, questions, or would like to contribute to the project, please feel free to raise an [issue](https://github.com/microsoft/horizon-benchmark/issues) or add a [pull request](https://github.com/microsoft/horizon-benchmark/pulls) 

---
## HORIZON Benchmark Construction

### a. Curating the Full Dataset:
The scripts for constructing the `HORIZON` benchmark are provided in the `data` folder. Follow the following steps to reproduce the benchmark:

1. Clone the repository:
```bash
git clone <REPO_NAME>
cd Horizon-Benchmark
```

2. Create and activate the environment from the YAML file:
```bash
conda env create -f environment.yaml
conda activate <your-env-name>
```
  
3. Give necessary permissions to the bash scripts:
```bash
chmod +x running_phase_one.sh
chmod +x running_phase_two.sh
chmod +x running_phase_three_metadata.sh
```

4. Run the bash scripts to curate the dataset files:
```bash
./running_phase_one.sh
./running_phase_two.sh
./running_phase_three_metadata.sh
```

Summary of the Bash Scripts:
- The process is extremely RAM-intensive due to the massive size of the corpus being created and the multiprocessing/batching optimizations performed to make it efficient. If your system doesnt support it, consider tweaking the hyperparameters.
- **Phase One** is the process of retrieving the category-wise data from the Amazon Reviews 2023 open-source repository and storing it in category-wise parquet files
- **Phase Two** is the process of merging these category-wise data to get a final `merged_user_all.json` file which contains the merged / category-agnostic user history of all the users in the Benchmark.
- **Phase Three** is the process of curating the metadata from Amazon Reviews and storing it in a `JSON`, `Parquet` and `DB` file. Necessary post-proc like filtering missing users/events and removing review texts is done to get lighter versions of the benchmark.
-  At the end of Phase Three, the following files shall be created:
    - `amazon_parquet_data/metadata_titles.db`: SQLite database containing 3 columns i.e. (1) Product ASIN, (2) Product title and (3) Product category for all items in the catalog
    - `amazon_parquet_data/merged_users_all_final_filtered.json`: Cleaned final full data JSON file with all users (removing those users with 0 lengths or missing titles in the metadata). The structure of this final JSON is as follows:
      ```bash
      {
        "{user_id}": {
              "{history}" : [I_1, I_2, .... , I_T],
              "{timestamps}" : [t_1, t_2, .... , t_T],
              "{ratings}" : [r_1, r_2, .... , r_T],
              "{reviews}" : [review_1, review_2, .... , review_T],      
      }
      ```
   - `amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json`: Cleaned final full data JSON file with all users (removing those users with 0 lengths or missing titles in the metadata) and no reviews. We provide a lighter version of the full data for those who do not plan to use the `reviews` field in their study.

### b. Preparing the Splits
The script for generating the splits as described in the paper are shared in the `splits` folder. Follow the steps below to generate the splits:
1. Generate user IDs for the 4 splits (as described in the paper) and save them in a txt file. You would need the `amazon_parquet_data/merged_users_all_final_filtered_no_reviews.json` file from the previous steps for this. A random seed of `42` is set for the sampling and the temporal thresholds for validation is set at `2019` and for test at `2020`:
```bash
python3 prepare_split_ids_full.py
```
This generates 4 txt files corresponding to in-distribution and out-of-distribution validation and test set users. 

2. Populate the JSON files with the complete data of each corresponding split in the same format as the JSON shared before:
```bash
python3 write_splits_to_jsons_full.py
```
This generates 4 JSON files corresponding to the in-distribution and out-of-distribution validation and test set users.

---
## Privacy Statement:
[Microsoft Privacy Statement](https://www.microsoft.com/en-us/privacy/privacystatement)
