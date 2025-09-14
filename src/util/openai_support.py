import re
from openai import OpenAI
import json
import httpx

keyy = """ YOUR KEY """

client = OpenAI(api_key=keyy, http_client=httpx.Client(verify=False))

llm_prompt_simplify = """
    Given a list of sentences, reformat each sentence to the simplest phrases that would distinguish it from the other examples. Here are some formatting rules to follow:\n
    1. If a sentence contains OBJECTS and ATTRIBUTES which belong to that object, the ATTRIBUTE must always come first. For example, given the sentence "A dog which is purple", reformat it to "A purple dog". 
    2. If a sentence contains NEGATION, the NEGATING TERM always comes before the OBJECT clause. For example, given "This image contains a chicken but a butterfly is absent", reformat it to "Chicken but no butterfly". \n
    3. If a sentence contains PREPOSITIONs, try your best to make sure that the OBJECTS the PREPOSITION is describing are immediately before and after the PREPOSITION. For example, given "A bug which is flying much farther up from the bench", reformat it to "A flying bug above a bench".\n
    4. Whenever possible, reformat VERBs to be ATTRIBUTES. For example, given "A dog dancing while his owner is jumping", reformat it to "Dancing dog and jumping owner".\n
    5. If two sentences are very close to each other, reduce them down to the salient components. For example, given the sentences ["Butterflies in the clouds, a cat squatting looking up at it, and a man standing behind the cat watching it, on the grass with a tree.", "Butterflies in the clouds, a cat squatting looking up at it, and a man sitting behind the cat watching it, on the grass with a tree."], return: {0: "A man standing", 1: "A man sitting"}. (Of course, if there are other sentences in the list that are similar, you may want to keep more details so that the sentences are still distinguishable.)\n
    Here are some more general examples. If given the following sentences: ["A desktop computer sitting on top of a gray oak table lights up the room" ,"A gray oak computer sitting on top of a desktop table lights up the room", "A kitchen has metal cabinets and black countertops with shiny lights on top.", "A kitchen has black cabinets and metal countertops with shiny lights on top."], return:
    {0: "Desktop computer top of gray oak table", 1: "Gray oak computer top of desktop table", 2: "Metal cabinets and black countertops", 3: "Black cabinets and metal countertops"}. \n\n
    As a rule, be AS CONCISE AS POSSIBLE. If any information is repeated and unnecessary to keep in order to distinguish that text prompt from the others, discard it.
    \n\n\n\nNow, reformat the following list of sentences and return the JSON output. Your answer MUST be a valid JSON array of strings. Do not include anything other than this json array in your answer.\n
"""

llm_prompt_functional = """
    You are given: A LOOKUP LIST of functional words (e.g., ["ABOVE", "BELOW", "INSIDE OF", "MANY", "SMALL", "NO"]). A list of SENTENCES to process.\n
    Definitions: Functional words include: \n
    (a) Prepositions (e.g., ABOVE, BELOW, INSIDE OF, ON, IN, NEAR, WITH, BESIDE,) or their synonyms. \n
    (b) Size/shape terms (e.g., SMALL, BIG). \n
    (c) Numerical terms (e.g., ONE, TWO, THREE, etc.). If a number is greater than 5 (e.g., SEVEN, 100), replace it with "MANY". Do not replace it if it is one, two, three, four, or five. \n
    (d) Negatory terms (e.g., NO, WITHOUT). \n
    Non-functional words: Do not include verbs, adjectives, or any nouns unrelated to the functional categories above. Examples of non-functional words include "jumping", "sleeping", "cat", "man", etc. \n
    These should not be added to the LOOKUP LIST, even if they appear in the sentences. Even if there are two sentences that are very similar, do not try to distinguish them by adding these verbs, adjectives, or nouns to the LOOKUP LIST. Any form of a verb, including present participles, may not go in the LOOKUP LIST, no matter how frequently it appears in the sentences. 
    \n
    Rules: For each sentence: Identify any functional words or synonyms (including numbers). If a functional word or one of its synonyms (by meaning) appears in the sentence and is already in the LOOKUP LIST, replace it in the sentence with the LOOKUP LIST key, surrounded by angle brackets (e.g., "A man close to a puppy" -> "A man <NEAR> a puppy"). If that functional word is not in the LOOKUP LIST (and it is truly functional by the above definition), add it to the LOOKUP LIST, then replace its appearance with that new all-caps key in angle brackets. \n
    Do not add duplicates to the LOOKUP LIST. Do not add verbs, adjectives, or any non-functional words to the LOOKUP LIST. Replace numbers greater than 5 with "MANY" (add "MANY" to the list if not already present). \n
    After processing all sentences, output exactly one JSON array containing two sub-arrays: The first sub-array: the UPDATED LOOKUP LIST (only functional words, no duplicates). The second sub-array: the FINAL TRANSFORMED SENTENCES (with functional words surrounded by < >). Not every sentence needs functional words. Provide no additional commentary or text besides this JSON structure. \n
    Example of the required output format: [ [ "ABOVE", "INSIDE OF", "MANY" , "RIGHT OF"], [ "A bird <ABOVE> a tree", "Fifteen dogs is <MANY> dogs" , "A sitting chicken is <INSIDE OF> a house"] ] \n
    Example of wrong output: [ [ "ABOVE", "INSIDE OF", "MANY" , "RIGHT OF", "SITTING", "DOGS", "HOUSE"], [ "A bird <ABOVE> a tree", "Fifteen dogs is <MANY> dogs" , "A sitting chicken is <INSIDE OF> a house"] ] 
    \n
    Before you return the output, CHECK THAT THE NEW LOOKUP LIST STILL CONTAINS ALL THE ELEMENTS OF THE PREVIOUS ONE. ALSO CHECK THAT THE LOOKUP LIST ONLY CONTAINS FUNCTIONAL WORDS. CHECK THAT YOU ARE NOT MISSING ANY FUNCTIONAL WORDS. \nNow, here is the input:\n\n

"""

llm_prompt_funct_clean = """
Definitions: Functional words include: \n
    (a) Prepositions (e.g., ABOVE, BELOW, INSIDE OF, ON, IN, NEAR, WITH, BESIDE,) or their synonyms. \n
    (b) Size/shape terms (e.g., SMALL, BIG). \n
    (c) Numerical terms (e.g., ONE, TWO, THREE, etc.). If a number is greater than 5 (e.g., SEVEN, 100), replace it with "MANY". Do not replace it if it is one, two, three, four, or five. \n
    (d) Negatory terms (e.g., NO, WITHOUT). \n 
Non-functional words: Do not include verbs, adjectives, or any nouns unrelated to the functional categories above. Examples of non-functional words include "jumping", "sleeping", "cat", "man", etc. \n Any form of a verb, including present participles, may not go in the LIST.
Remove all words that are NOT functional words in this list, while maintaining the order. Return the answer as a single JSON array. Do not include anything other than the JSON array in your response:\n\n
"""


def simplify_prompt(system_prompt, texts, model="gpt-4o-mini", temperature=0.7):
    # Build the user message with enumerated lines
    user_lines = []
    for i, item in enumerate(texts, start=1):
        # Here, we expect the model to produce one JSON object for each line
        # so line #i corresponds to the i-th item in the output array.
        user_lines.append(f"{i}) {item}")
    user_message = "\n".join(user_lines)
    assistant_text = None

    # Because your system prompt already says "Please produce a single JSON array..."
    # we simply provide that as the system content and enumerated lines as user content.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1500,
            top_p=1,
        )
        # Access content in the new library:
        assistant_text = None
        assistant_text = response.choices[0].message.content.strip()
        # print("Raw response:", assistant_text)
        # Attempt to parse the entire text as a single JSON array
        # (One element per input line)
        match = re.search(r"(\[.*\])", assistant_text, re.DOTALL)
        if match:
            assistant_text = match.group(1)
        parsed = json.loads(assistant_text)

        print("----")
        if not isinstance(parsed, list):
            print("Warning: Model did not return a list. Returning empty.")
            return []
        return parsed
    except Exception as e:
        if assistant_text:
            print(assistant_text)
        else:
            print("No assistant text returned at all")
        print(f"Error in simplify_prompt: {e}")
        return []


def find_functional_words(
    system_prompt, texts, lookup_list, model="gpt-4o-mini", temperature=0.7
):
    # Build the user message with enumerated lines
    user_lines = (
        "LOOKUP LIST: ["
        + ", ".join(lookup_list)
        + "]\nSENTENCES: ["
        + ",\n".join(texts)
        + "]"
    )

    user_message = user_lines
    assistant_text = None

    # Because your system prompt already says "Please produce a single JSON array..."
    # we simply provide that as the system content and enumerated lines as user content.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1500,
            top_p=1,
        )
        # Access content in the new library:
        assistant_text = None
        assistant_text = response.choices[0].message.content.strip()
        # print("Raw response:", assistant_text)
        # Attempt to parse the entire text as a single JSON array
        # (One element per input line)
        match = re.search(
            r"\[\s*(\[[^\[\]]*\])\s*,\s*(\[[^\[\]]*\])\s*\]", assistant_text, re.DOTALL
        )

        if match:
            # Extract the two arrays from the match groups
            ar1_str = match.group(1)
            ar2_str = match.group(2)

            # Parse each array
            ar1 = json.loads(ar1_str)
            ar2 = json.loads(ar2_str)

        else:
            ar1 = lookup_list
            ar2 = texts

        print("----")

        if not isinstance(ar1, list):
            print("Warning: Lookup list not a list. Returning empty.")
            return [], []
        if not isinstance(ar2, list):
            print("Warning: sentences are not a list. Returning empty.")
            return [], []
        return ar1, ar2  # first one is lookup list, second is sentences.
    except Exception as e:
        if assistant_text:
            print("assistant text:", assistant_text)
        else:
            print("No assistant text returned at all")
        print(f"Error in finding functional words: {e}")
        return [], []


def clean_funct_words(system_prompt, lookup_list, model="gpt-4o-mini", temperature=0.7):
    user_message = "[" + ", ".join(lookup_list) + "]"
    assistant_text = None

    # Because system prompt already says "Please produce a single JSON array..."
    # we simply provide that as the system content and enumerated lines as user content.
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=1500,
            top_p=1,
        )
        # Access content in the new library:
        assistant_text = None
        assistant_text = response.choices[0].message.content.strip()
        # print("Raw response:", assistant_text)
        # Attempt to parse the entire text as a single JSON array
        # (One element per input line)
        match = re.search(r"(\[.*\])", assistant_text, re.DOTALL)
        if match:
            assistant_text = match.group(1)
            parsed = json.loads(assistant_text)
        else:
            parsed = lookup_list

        print("----")
        if not isinstance(parsed, list):
            print("Warning: Model did not return a list. Returning empty.")
            return []
        return parsed
    except Exception as e:
        if assistant_text:
            print(assistant_text)
        else:
            print("No assistant text returned at all")
        print(f"Error in cleaning funct words: {e}")
        return []
