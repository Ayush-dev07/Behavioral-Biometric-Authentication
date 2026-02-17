def update_template(template, new_emb, alpha=0.1):
    return (1 - alpha) * template + alpha * new_emb
