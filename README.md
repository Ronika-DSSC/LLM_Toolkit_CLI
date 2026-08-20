# Create a conda environment to install all the required packages.
# Create a .env file with the LLM and EMBEDDING URLs and API keys.

source <path_to_conda.sh>
conda activate llm_toolkit

python LLM_Toolkit_CLI.py   --input-folder ./UD7/Medical_Record \
                            --markdown-dir ./data/markdown \
                            --folder ./data/markdown \
                            --backend chroma \
                            --vector-store-dir ./my_rag_datastore \
                            --collection-name clinical_chromaDB_test_docs \
                            --max-tokens 400 \
                            --overlap-tokens 40 \
                            --top-k 5 \
                            --cosine-threshold 0.3 \
                            --prompt-file ./prompts/promptset/clinical_characteristics_hlh_diagnosis_prompts_v3_with_page_evidence.txt \
                            --prompt-index 0 \
                            --llm-model phi4:14b \
                            --temperature 0.1 \
                            --verbose
