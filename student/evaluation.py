import json
from models import RagDataset, StudentSearchResults

def calculate_overlap_percentage(s_truth: int, e_truth: int, s_retrieve: int, e_retrieve: int) -> float:
    """Calculate how much of the truth is covered by the retrieved chunk."""
    overlap_start = max(s_truth, s_retrieve)
    overlap_end = min(e_truth, e_retrieve)
    overlap_size = max(0, overlap_end - overlap_start)

    truth_size = max(1, e_truth - s_truth) # Avoid division by zero
    
    return overlap_size / truth_size

def evaluate_results(student_answer_path: str, dataset_path: str, k: int = 10):
    try:
        with open(student_answer_path, "r") as f:
            student_data = StudentSearchResults.model_validate_json(f.read())
            
        with open(dataset_path, "r") as f:
            truth_data = RagDataset.model_validate_json(f.read())

    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # FIX 1: Map by the actual question string, NOT the randomly generated question_id!
    truth_map = {q.question_id: q for q in truth_data.rag_questions if hasattr(q, 'sources')}
    
    total_questions = len(truth_map)
    if total_questions == 0:
        print("No answered questions found in dataset.")
        return
        
    print(f"Student data is valid: True")
    print(f"Total number of questions: {len(truth_data.rag_questions)}")
    print(f"Total number of questions with sources: {total_questions}")
    print(f"Total number of questions with student sources: {len(student_data.search_results)}")
    
    # Calculate recall at different K values
    k_values = [1, 3, 5, 10]
    recalls = {kval: 0.0 for kval in k_values}
    
    matched_qs = 0
    for result in student_data.search_results:
        # Check against the string to bypass UUID mismatches
        if result.question_id not in truth_map:
            continue
            
        matched_qs += 1
        truth_question = truth_map[result.question_id]
        truth_sources = truth_question.sources
        
        for kval in k_values:
            top_k_retrieved = result.retrieved_sources[:kval]
            found_count = 0
            
            for gt in truth_sources:
                found = False
                for retrieved in top_k_retrieved:
                    # FIX 2: Safer path comparison (ignores leading slashes or repo prefixes)
                    if gt.file_path.endswith(retrieved.file_path) or retrieved.file_path.endswith(gt.file_path):
                        overlap = calculate_overlap_percentage(
                            gt.first_character_index, gt.last_character_index,
                            retrieved.first_character_index, retrieved.last_character_index
                        )
                        if overlap >= 0.05: # 5% minimum overlap rule
                            found = True
                            break
                if found:
                    found_count += 1
                    
            score = found_count / len(truth_sources)
            recalls[kval] += score
            
    print("\nEvaluation Results")
    print("========================================")
    print(f"Questions evaluated: {matched_qs} / {total_questions}")
    for kval in k_values:
        if kval <= k:
            avg_recall = recalls[kval] / matched_qs if matched_qs > 0 else 0.0
            print(f"Recall@{kval}: {avg_recall * 100}")
