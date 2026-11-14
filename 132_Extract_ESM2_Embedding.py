# 参考：https://github.com/facebookresearch/ESM#getting-started-with-this-repo-

# 1. 简单封装
def get_esm2_embeddings(sequence_label:str, sequences: str, model_name: str):
    import torch 
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    
    data = [(sequence_label, sequences)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    
    # mean pooling
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1:tokens_len - 1].mean(0))

    # or sequence_representations[0].numpy()
    return sequence_representations[0]

############################################################################################################


# 2. idpfold2
# 参考：https://github.com/Junjie-Zhu/IDPFold2/blob/be66d6ea280e140c87eeac61d6c7d4b33230c310/scripts/get_esm_embedding.py
# bulk用

def calculate_representation(model, alphabet, data, device, output_dir):
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    total_sequences = len(data)
    num_batches = total_sequences // BATCH_SIZE + (total_sequences % BATCH_SIZE != 0)

    for batch in tqdm(range(num_batches)):
        start_idx, end_idx = batch * BATCH_SIZE, (batch + 1) * BATCH_SIZE
        batch_items = data[start_idx:end_idx]
        converter_input = [(names[0], sequence) for names, sequence in batch_items]
        _, _, batch_tokens = batch_converter(converter_input)
        batch_tokens = batch_tokens.to(device)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            token_representations = model(batch_tokens, repr_layers=[33], return_contacts=True)['representations'][33]

        for i, tokens_len in enumerate(batch_lens):
            embedding = token_representations[i, 1: tokens_len - 1].cpu()
            names = batch_items[i][0]
            primary_path = embedding_path(output_dir, names[0])
            torch.save(embedding, primary_path)
            for name in names[1:]:
                dest_path = embedding_path(output_dir, name)
                if os.path.abspath(dest_path) != os.path.abspath(primary_path):
                    shutil.copyfile(primary_path, dest_path)

        del batch_tokens, token_representations
        if device == 'cuda':
            torch.cuda.empty_cache()





