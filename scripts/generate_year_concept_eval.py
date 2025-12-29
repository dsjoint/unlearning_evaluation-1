#!/usr/bin/env python3
"""
Generate deterministic year concept evaluation dataset.

Creates MCQ questions testing general year understanding:
- Year ordering (which is earlier/later)
- Year arithmetic (addition/subtraction)
- Decade/century classification

Output: data/year_concept_eval/year_concept.jsonl
Format: JSONL with {"question": str, "choices": [str, str, str, str], "answer": int}
"""

import json
import random
import os
from pathlib import Path

# Fixed seed for reproducibility
RANDOM_SEED = 42
NUM_ORDERING = 100
NUM_ARITHMETIC = 100
NUM_CLASSIFICATION = 100
NUM_BOUNDARY = 100

def generate_ordering_questions(rng: random.Random) -> list[dict]:
    """Generate year ordering questions: which is earlier/later."""
    questions = []
    for _ in range(NUM_ORDERING):
        # Sample two years from 1800-2020
        year1 = rng.randint(1800, 2020)
        year2 = rng.randint(1800, 2020)
        
        # Ensure they're different
        while year2 == year1:
            year2 = rng.randint(1800, 2020)
        
        # Randomly choose "earlier" or "later"
        is_earlier = rng.choice([True, False])
        
        if is_earlier:
            question = f"Which year is earlier: {year1} or {year2}?"
            correct_year = min(year1, year2)
        else:
            question = f"Which year is later: {year1} or {year2}?"
            correct_year = max(year1, year2)
        
        # Create 4 choices: correct answer + 3 distractors
        choices = [str(correct_year)]
        # Add distractors: years near the correct answer
        max_attempts = 100
        for _ in range(3):
            attempts = 0
            distractor = rng.randint(correct_year - 10, correct_year + 10)
            while (distractor == correct_year or str(distractor) in choices) and attempts < max_attempts:
                distractor = rng.randint(correct_year - 10, correct_year + 10)
                attempts += 1
            if attempts < max_attempts:
                choices.append(str(distractor))
            else:
                # Fallback: use a year further away
                distractor = rng.randint(1800, 2020)
                while str(distractor) in choices:
                    distractor = rng.randint(1800, 2020)
                choices.append(str(distractor))
        
        # Shuffle choices
        rng.shuffle(choices)
        answer = choices.index(str(correct_year))
        
        questions.append({
            "question": question,
            "choices": choices,
            "answer": answer
        })
    
    return questions

def generate_arithmetic_questions(rng: random.Random) -> list[dict]:
    """Generate year arithmetic questions: addition/subtraction."""
    questions = []
    for _ in range(NUM_ARITHMETIC):
        # Base year from 1800-2020
        base_year = rng.randint(1800, 2020)
        # Offset: -50 to +50 years
        offset = rng.randint(-50, 50)
        while offset == 0:
            offset = rng.randint(-50, 50)
        
        result_year = base_year + offset
        
        # Randomly choose addition or subtraction phrasing
        if rng.choice([True, False]):
            if offset > 0:
                question = f"What year is {offset} years after {base_year}?"
            else:
                question = f"What year is {abs(offset)} years before {base_year}?"
        else:
            question = f"{base_year} {'plus' if offset > 0 else 'minus'} {abs(offset)} years equals what year?"
        
        # Create 4 choices: correct answer + 3 distractors
        choices = [str(result_year)]
        # Add distractors: years near the result
        max_attempts = 100
        for _ in range(3):
            attempts = 0
            distractor = rng.randint(result_year - 10, result_year + 10)
            while (distractor == result_year or str(distractor) in choices) and attempts < max_attempts:
                distractor = rng.randint(result_year - 10, result_year + 10)
                attempts += 1
            if attempts < max_attempts:
                choices.append(str(distractor))
            else:
                # Fallback: use a year further away
                distractor = rng.randint(1800, 2020)
                while str(distractor) in choices:
                    distractor = rng.randint(1800, 2020)
                choices.append(str(distractor))
        
        # Shuffle choices
        rng.shuffle(choices)
        answer = choices.index(str(result_year))
        
        questions.append({
            "question": question,
            "choices": choices,
            "answer": answer
        })
    
    return questions

def generate_classification_questions(rng: random.Random) -> list[dict]:
    """Generate decade/century classification questions."""
    questions = []
    for _ in range(NUM_CLASSIFICATION):
        year = rng.randint(1800, 2020)
        
        # Randomly choose decade or century
        question_type = rng.choice(["decade", "century"])
        
        if question_type == "decade":
            # Decade: e.g., 1978 -> 1970s
            decade_start = (year // 10) * 10
            question = f"Which decade is {year} in?"
            correct_answer = f"{decade_start}s"
            
            # Create choices: correct decade + 3 nearby decades
            choices = [correct_answer]
            valid_offsets = [-30, -20, -10, 10, 20, 30]
            for _ in range(3):
                rng.shuffle(valid_offsets)
                added = False
                for offset in valid_offsets:
                    distractor_decade = decade_start + offset
                    if distractor_decade >= 1800 and distractor_decade <= 2020:
                        distractor = f"{distractor_decade}s"
                        if distractor not in choices:
                            choices.append(distractor)
                            added = True
                            break
                if not added:
                    # Fallback: use a random decade in range
                    distractor_decade = rng.randint(1800, 2020) // 10 * 10
                    distractor = f"{distractor_decade}s"
                    max_attempts_fallback = 100
                    attempts_fallback = 0
                    while distractor in choices and attempts_fallback < max_attempts_fallback:
                        distractor_decade = rng.randint(1800, 2020) // 10 * 10
                        distractor = f"{distractor_decade}s"
                        attempts_fallback += 1
                    if distractor not in choices:
                        choices.append(distractor)
            
            # Fill remaining slots if needed
            max_attempts = 100
            attempts = 0
            while len(choices) < 4 and attempts < max_attempts:
                offset = rng.choice([-30, -20, -10, 10, 20, 30])
                distractor_decade = decade_start + offset
                if distractor_decade >= 1800 and distractor_decade <= 2020:
                    distractor = f"{distractor_decade}s"
                    if distractor not in choices:
                        choices.append(distractor)
                attempts += 1
            
            # Final fallback if still not enough choices
            while len(choices) < 4:
                distractor_decade = rng.randint(1800, 2020) // 10 * 10
                distractor = f"{distractor_decade}s"
                if distractor not in choices:
                    choices.append(distractor)
        
        else:  # century
            # Century: e.g., 1905 -> 20th century (1901-2000)
            if year >= 1901:
                century_num = ((year - 1) // 100) + 1
            else:
                century_num = (year // 100) + 1
            question = f"Which century is {year} in?"
            correct_answer = f"{century_num}{get_ordinal_suffix(century_num)} century"
            
            # Create choices: correct century + 3 nearby centuries
            choices = [correct_answer]
            valid_offsets = [-2, -1, 1, 2]
            for _ in range(3):
                rng.shuffle(valid_offsets)
                added = False
                for offset in valid_offsets:
                    distractor_century = century_num + offset
                    if distractor_century >= 18 and distractor_century <= 21:
                        distractor = f"{distractor_century}{get_ordinal_suffix(distractor_century)} century"
                        if distractor not in choices:
                            choices.append(distractor)
                            added = True
                            break
                if not added:
                    # Fallback: use a random valid century
                    distractor_century = rng.choice([18, 19, 20, 21])
                    distractor = f"{distractor_century}{get_ordinal_suffix(distractor_century)} century"
                    max_attempts_fallback = 100
                    attempts_fallback = 0
                    while distractor in choices and attempts_fallback < max_attempts_fallback:
                        distractor_century = rng.choice([18, 19, 20, 21])
                        distractor = f"{distractor_century}{get_ordinal_suffix(distractor_century)} century"
                        attempts_fallback += 1
                    if distractor not in choices:
                        choices.append(distractor)
            
            # Fill remaining slots if needed
            max_attempts = 100
            attempts = 0
            while len(choices) < 4 and attempts < max_attempts:
                offset = rng.choice([-2, -1, 1, 2])
                distractor_century = century_num + offset
                if distractor_century >= 18 and distractor_century <= 21:
                    distractor = f"{distractor_century}{get_ordinal_suffix(distractor_century)} century"
                    if distractor not in choices:
                        choices.append(distractor)
                attempts += 1
            
            # Final fallback if still not enough choices
            while len(choices) < 4:
                distractor_century = rng.choice([18, 19, 20, 21])
                distractor = f"{distractor_century}{get_ordinal_suffix(distractor_century)} century"
                if distractor not in choices:
                    choices.append(distractor)
        
        # Shuffle choices
        rng.shuffle(choices)
        answer = choices.index(correct_answer)
        
        questions.append({
            "question": question,
            "choices": choices,
            "answer": answer
        })
    
    return questions

def generate_boundary_questions(rng: random.Random) -> list[dict]:
    """Generate year vs non-year boundary questions: distinguish calendar years from decimals/dates/IDs/versions."""
    questions = []
    for _ in range(NUM_BOUNDARY):
        # Generate a valid calendar year (1800-2020)
        correct_year = rng.randint(1800, 2020)
        correct_year_str = str(correct_year)
        
        # Create distractors based on different patterns
        distractor_types = [
            "decimal",      # e.g., 3.14, 19.99
            "date",         # e.g., 12/31, 01/01
            "version",      # e.g., v2.0, v1.5
            "id",           # e.g., ID123, 1999A
            "malformed",    # e.g., 200O, 2OO1, 200I, 18999, 199A, 19 99
        ]
        
        # Randomly select 3 distractor types
        selected_types = rng.sample(distractor_types, 3)
        choices = [correct_year_str]
        
        for dist_type in selected_types:
            if dist_type == "decimal":
                # Decimal number
                if rng.choice([True, False]):
                    # Pi-like: 3.14
                    distractor = f"{rng.randint(1, 5)}.{rng.randint(10, 99)}"
                else:
                    # Year-like decimal: 19.99, 20.20
                    distractor = f"{rng.randint(18, 21)}.{rng.randint(0, 99):02d}"
            
            elif dist_type == "date":
                # Date format: MM/DD or DD/MM
                if rng.choice([True, False]):
                    distractor = f"{rng.randint(1, 12):02d}/{rng.randint(1, 31):02d}"
                else:
                    distractor = f"{rng.randint(1, 31):02d}/{rng.randint(1, 12):02d}"
            
            elif dist_type == "version":
                # Version number: v2.0, v1.5, version 3.1
                if rng.choice([True, False]):
                    distractor = f"v{rng.randint(1, 5)}.{rng.randint(0, 9)}"
                else:
                    distractor = f"version {rng.randint(1, 5)}.{rng.randint(0, 9)}"
            
            elif dist_type == "id":
                # ID format: 1999A, ID123, 2001-X
                id_formats = [
                    f"{rng.randint(1800, 2020)}{rng.choice(['A', 'B', 'C', 'X'])}",
                    f"ID{rng.randint(100, 9999)}",
                    f"{rng.randint(1800, 2020)}-{rng.choice(['X', 'A', 'B'])}",
                ]
                distractor = rng.choice(id_formats)
            
            elif dist_type == "malformed":
                # Malformed years: character substitutions or invalid formats
                malformed_types = [
                    # Character substitution: O instead of 0, I instead of 1
                    correct_year_str.replace("0", "O").replace("1", "I"),
                    correct_year_str.replace("0", "O"),
                    correct_year_str.replace("1", "I"),
                    # Too many digits: 18999, 20000
                    f"{rng.randint(18000, 99999)}",
                    # Too few digits: 99, 999
                    f"{rng.randint(10, 999)}",
                    # Space in year: 19 99, 20 01
                    f"{correct_year_str[:2]} {correct_year_str[2:]}",
                    # Letter in middle: 199A, 20B1
                    f"{correct_year_str[:-1]}{rng.choice(['A', 'B', 'X'])}",
                ]
                distractor = rng.choice(malformed_types)
                # Ensure it's different from correct year
                while distractor == correct_year_str or distractor in choices:
                    distractor = rng.choice(malformed_types)
            
            # Ensure distractor is unique
            if distractor not in choices:
                choices.append(distractor)
        
        # If we don't have 4 choices yet, add more distractors
        while len(choices) < 4:
            # Add a random distractor from any type
            dist_type = rng.choice(distractor_types)
            if dist_type == "decimal":
                distractor = f"{rng.randint(1, 5)}.{rng.randint(10, 99)}"
            elif dist_type == "date":
                distractor = f"{rng.randint(1, 12):02d}/{rng.randint(1, 31):02d}"
            elif dist_type == "version":
                distractor = f"v{rng.randint(1, 5)}.{rng.randint(0, 9)}"
            elif dist_type == "id":
                distractor = f"ID{rng.randint(100, 9999)}"
            else:  # malformed
                distractor = f"{rng.randint(18000, 99999)}"
            
            if distractor not in choices:
                choices.append(distractor)
        
        # Shuffle choices
        rng.shuffle(choices)
        answer = choices.index(correct_year_str)
        
        questions.append({
            "question": "Which option is most plausibly a calendar year (not a decimal, date, or ID)?",
            "choices": choices,
            "answer": answer,
            "correct_year": correct_year  # Store for reference
        })
    
    return questions

def get_ordinal_suffix(n: int) -> str:
    """Get ordinal suffix: 1st, 2nd, 3rd, 4th, etc."""
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

def main():
    # Set seed for reproducibility
    rng = random.Random(RANDOM_SEED)
    
    # Generate all question types
    ordering_questions = generate_ordering_questions(rng)
    arithmetic_questions = generate_arithmetic_questions(rng)
    classification_questions = generate_classification_questions(rng)
    boundary_questions = generate_boundary_questions(rng)
    
    # Combine all questions
    all_questions = ordering_questions + arithmetic_questions + classification_questions + boundary_questions
    
    # Create output directory
    output_dir = Path("data/year_concept_eval")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write to JSONL file
    output_file = output_dir / "year_concept.jsonl"
    with open(output_file, 'w') as f:
        for q in all_questions:
            f.write(json.dumps(q) + '\n')
    
    print(f"Generated {len(all_questions)} year concept evaluation questions")
    print(f"  - Ordering: {len(ordering_questions)}")
    print(f"  - Arithmetic: {len(arithmetic_questions)}")
    print(f"  - Classification: {len(classification_questions)}")
    print(f"  - Boundary: {len(boundary_questions)}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()

