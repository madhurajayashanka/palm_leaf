import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from pipeline import load_knowledge_graph, analyze_safety

def run_cascading_failure_test():
    kg = load_knowledge_graph()

    print("=== Cascading Failure Error Propagation ===")
    print("Hypothesis: Incorrect sentence boundary detection (SBD) leads to Safety Guardrail failures.\n")

    scenarios = [
        {
            "name": "1. Perfect Segmentation (SBD Accuracy: 100%)",
            "text": "වාත රෝග සඳහා නියඟලා අලයක් ගෙන හොඳින් සුද්ද කරගන්න. ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න. පසුව වේලා කුඩු කරගන්න.",
            "expected": "APPROVED"
        },
        {
            "name": "2. Mild Over-segmentation (SBD Accuracy: ~80%)",
            "text": "වාත රෝග සඳහා නියඟලා අලයක් ගෙන. හොඳින් සුද්ද කරගන්න. ඉන්පසු එය ගොම දියරේ දින තුනක් ගිල්වා තබන්න. පසුව වේලා කුඩු කරගන්න.",
            "expected": "APPROVED"
        },
        {
            "name": "3. Severe Over-segmentation (SBD Accuracy: ~50%) - The Cascading Failure",
            "text": "වාත රෝග සඳහා නියඟලා අලයක් ගෙන. හොඳින් සුද්ද. කරගන්න ඉන්පසු. එය ගොම දියරේ. දින තුනක් ගිල්වා තබන්න. පසුව වේලා කුඩු කරගන්න.",
            "expected": "REJECTED" 
        }
    ]

    window_size = 1
    print(f"Safety Window Size (k) = {window_size}\n")

    for s in scenarios:
        res = analyze_safety(s["text"], kg, window_size=window_size)
        status = res["final_status"]
        print(f"Scenario: {s['name']}")
        print(f"Segmented Text: {s['text']}")
        print(f"Guardrail Verdict: {status}")
        
        if status == "REJECTED":
            print("-> 🔴 CASCADING FAILURE TRIGGERED: Safe recipe was incorrectly rejected due to bad SBD pushing the purification keywords out of context.")
        else:
            print("-> 🟢 PASS: Context window successfully captured purification keywords.")
        print("-" * 50)

if __name__ == "__main__":
    run_cascading_failure_test()
