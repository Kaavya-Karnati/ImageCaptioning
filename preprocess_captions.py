import pandas as pd

def preprocess_captions(input_file, output_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            parts = line.split(',')  # Split by comma
            if len(parts) != 2:  # Skip malformed lines
                print(f"Skipping malformed line: {line}")
                continue
            
            image_id, caption = parts
            image_id = image_id.split('#')[0]  # Remove the #0, #1, etc.
            data.append((image_id, caption))

    # Save the processed data to a CSV file
    df = pd.DataFrame(data, columns=["image", "caption"])
    df.to_csv(output_file, index=False)
    print(f"Preprocessed captions saved to {output_file}")

# Run the function
preprocess_captions("data/captions.txt", "data/captions.csv")
