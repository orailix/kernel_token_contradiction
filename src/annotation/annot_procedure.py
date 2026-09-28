ANNOTATION_INSTRUCTIONS = """
# Task

Given the original **question**, the model's **answer**, a **segmented version** of that
answer, and some **reference knowledge**, assign a factuality score to each segment.

# Scoring Convention

-1 → Factually incorrect / contradicts previous ideas / unverifiable / approximative
1 → Factually correct

Be **tough and rigorous**: Only assign "1" when the content is clearly factually correct
and relevant to the question.


# Edge Cases

1. **Denial of answer:**
If the model explicitly states it cannot answer or denies having information
(e.g., "I did not find any information..."), assign label 1.
The denial itself is factually correct.

2. **Targeted labels:**
When a segment is incorrect, apply the label to the specific factual claim, not the surrounding phrasing.
Example: Question = "Who was the first Carolingian king?"
Answer = "The first Carolingian king was Clovis."
Here, the incorrect part is "Clovis," not the framing "The first Carolingian...".

3. **Numbers:**
   3a. *Exact values required* (e.g., dates, rank in a competition, small quantities <100):
       - Exact match → 1
       - Otherwise → -1
   3b. *Approximate values acceptable* (e.g., population, area):
       - Exact match → 1
       - Within 10% error with "around", "approximately" or similar context → 1
       - Within 10% error without such context → -1
       - Larger error → -1

4. **Chunk granularity:**
If a chunk contains any factually false statement, the entire chunk should be labeled -1.

5. **Overgeneration mistakes and false details:**
If the answer contains overgeneration that adds details to an incorrect claim (e.g., stating the citizenship
or birthdate of the wrong person, or providing details about an event that did not occur), assign -1 to both
the incorrect claim and any overgenerated or unverifiable details associated with it.

6. **EOS token**
End-of-sequence tokens such as "<eos>", "</s>" or "<|eot_id|>" should be labeled 1, except if they appear
in the same chunk as incorrect information (see rule 4).
"""

RAG_CHUNK_SIZE = 120
RAG_TOP_K = 3
