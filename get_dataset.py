from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("graelo/wikipedia", "20230601.es")

    with open ('wiki.txt', 'r') as file:
        urls = [line.strip() for line in file.readlines()]
        urls = set(urls)
    urls
    indices = []

    for i, url in enumerate(dataset['train']['url']):
        if url in urls:
            indices.append(i)

    filtered_dataset = dataset['train'].select(indices)
    
    return filtered_dataset

if __name__ == "__main__":
    dataset = get_dataset()

