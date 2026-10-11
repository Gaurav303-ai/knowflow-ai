import tiktoken

# Create tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-4o")

# Take user input
text = input("Enter your text: ")

# Convert text to tokens
token_ids = tokenizer.encode(text)

# Print results
print("\nOriginal Text :", text)
print("Total Words :", len(text.split()))
print("Token IDs     :", token_ids)
print("Total Tokens  :", len(token_ids))
print("Decoded Text  :", tokenizer.decode(token_ids))
