import PyPDF2
from collections import Counter
import tiktoken

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip]
        preprocessed = [item if item in self.str_to_int  #A
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) #B
        return text

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# === 2. Tokenize the Text using GPT-2 BPE tokenizer ===
def tokenize_with_tiktoken(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    decoded_tokens = [tokenizer.decode([t]) for t in token_ids]
    counter = Counter(decoded_tokens)
    return token_ids, counter.most_common(50)

# === 3. Run Everything ===
pdf_path = "text_data/Liane_M_Summerfield_Nutrition_Exercise_and_Behavior_An_Integrated_Approach_to_Weight_Management_Second_edition.pdf"  # 🔁 Change this to your actual file path
raw_text = extract_text_from_pdf(pdf_path)

print(f"Total characters in extracted text: {len(raw_text)}")

token_ids, common_words = tokenize_with_tiktoken(raw_text)
print(f"Total tokens: {len(token_ids)}")
print("First 50 tokens (IDs):", token_ids[:50])
print("First 50 words:", common_words[:50])
