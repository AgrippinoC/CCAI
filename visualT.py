import cv2, torch, time, asyncio
from dotenv import load_dotenv

import numpy as np
import matplotlib.pyplot as plt
from datasets import DatasetDict, load_dataset
from evaluate import load
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

load_dotenv()

feature_extractor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

#carico modello e dataset
def model_e_dataset():
    ds_totale = load_dataset("imagefolder", data_dir = "./ImperImg")
    t_set  = ds_totale["train"].train_test_split(test_size = 0.3, seed = 40)
    v_set = t_set["test"].train_test_split(test_size = 0.5, seed = 41)

    ds = DatasetDict({
        "train": t_set["train"],
        "validation": v_set["train"],
        "test": v_set["test"]
    })

    name_labels = ds['train'].features['label'].names
    num_labels = len(name_labels)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        id2label={str(i): c for i, c in enumerate(name_labels)},
        label2id={c: i for i, c in enumerate(name_labels)},
        attn_implementation="eager",
        output_attentions=False,
        ignore_mismatched_sizes=True,
    )

    return ds, model

#preprocessing
def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs
def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

#metriche
accuracy = load("accuracy")
precision = load("precision")
recall = load("recall")
f1score = load("f1")
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    labels = p.label_ids

    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "precision": precision.compute(predictions=preds, references=labels, average="micro")["precision"],
        "recall": recall.compute(predictions=preds, references=labels, average="micro")["recall"],
        "f1-score": f1score.compute(predictions=preds, references=labels, average="micro")["f1"]
    }

#mappa di salienza e predizione
def attention(model, pil, title=""):
    """Prende un modelo e un'immagine PIL e visualizza l'attention map sovrapposta."""
    model.eval()

    #Preprocessing
    inputs = feature_extractor(images=pil, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    #Inferenza
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attentions = outputs.attentions

    att_mat = torch.stack(attentions).squeeze(1)
    att_mat = torch.mean(att_mat, dim=1)

    #Connessioni residue e normalizzazione
    residual_att = torch.eye(att_mat.size(1)).to(device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    #Rollout
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    #Attenzione dal CLS allo spazio di input
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))

    #Mask
    mask = v[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    mask = cv2.resize(mask / mask.max(), pil.size)

    #Plot finale
    image_np = np.array(pil.convert("RGB"))
    plt.figure(figsize=(6, 6))
    plt.imshow(image_np)
    plt.imshow(mask, cmap="jet", alpha=0.45)
    plt.title(f"Attention Map - {title}")
    plt.axis("off")
    plt.show()
def prediction(model, pil):
    """Prende un modelo e un'immagine PIL, mostra le top predizioni"""
    model.eval()

    #Preprocessing img
    inputs = feature_extractor(images=pil, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    #Inferenza
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    logits = outputs.logits

    labels_map = model.config.id2label
    probs = torch.nn.Softmax(dim=-1)(logits)
    top_k = min(5, len(labels_map))
    top_predictions = torch.argsort(probs, dim=-1, descending=True)

    print(f"\nPREDIZIONI")
    for idx in top_predictions[0, :top_k]:
        label_id = idx.item()
        print(f'{probs[0, label_id]:.5f} : {labels_map[str(label_id)]}')

#modellazione
def experiment(model, trans_ds, lr, ep, size, exp_name, lora):
    if lora:
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.3,
            bias="none",
            modules_to_save=["classifier"],
        )
        model = get_peft_model(model, config)

    training_args = TrainingArguments(
        output_dir=f"./vit-{exp_name}",
        per_device_train_batch_size=size,
        num_train_epochs=ep,
        learning_rate=lr,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=trans_ds["train"],
        eval_dataset=trans_ds["validation"],
        processing_class=feature_extractor,
        )

    trainer.train()
    metrics = trainer.evaluate(trans_ds["validation"])

    print(f"\nESPERIMENTO {exp_name}\n")
    print(metrics)

    return metrics, lr, ep, size, trainer

def build_model():
    ds, _ = model_e_dataset()

    name_labels = ds['train'].features['label'].names
    num_labels = len(name_labels)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=num_labels,
        id2label={str(i): c for i, c in enumerate(name_labels)},
        label2id={c: i for i, c in enumerate(name_labels)},
        attn_implementation="eager",
        ignore_mismatched_sizes=True,
    )

    return ds, model

async def main():

    ds, mod = model_e_dataset()
    trans_ds = ds.with_transform(transform)
    img_test = ds["test"][9]["image"]

    #1) MODELLO BASELINE
    time.sleep(2)
    print("\nTest 1 modello non fine-tunato su un' img a caso")
    attention(mod, img_test, title="NON FINE TUNING")
    prediction(mod, img_test)
    
    #momentaneo
    trainer_baseline = Trainer(
        model=mod,
        args=TrainingArguments(output_dir="./tmp", fp16=torch.cuda.is_available(), remove_unused_columns=False),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    metrics_baseline = trainer_baseline.evaluate(trans_ds['test'])
    print("\nLog delle metriche pre training sul test-dataset")
    trainer_baseline.log_metrics("test", metrics_baseline)

    # FASE 2) FINE-TUNING
    time.sleep(10)

    # HYPERPARAMETER SEARCH
    exp = [(3e-4, 1, 8, "N1"), (1e-4, 2, 4, "N2"), (3e-5, 4, 4, "N3"),]
    results = []

    for lr, ep, bs, name in exp:
        print(f"\nPROVA {name}")
        _, model = build_model()
        metrics,lerng,epo,siz, _ = experiment(model, trans_ds, lr, ep, bs, name, False)
        results.append((name,lerng,epo,siz, metrics))
        time.sleep(3)

    print("\nRISULTATI\n")
    mas = 0.0
    le = 0.0
    eposd = 0.0
    bsrt = 0.0
    n = ""
    for name, l, e, s, m in results:
        mm = m["eval_f1-score"]
        print(name, "-->", m, "\n")
        if mm > mas:
            mas = mm
            le = l
            eposd = e
            bsrt = s
            n = name

    print(f"Il miglior modello 'Full-Fine-Tuned' = {n}\n")
    
    # TRAINING LoRa
    _, best_model = build_model()
    time.sleep(5)
    print(f"PROVA CON LORA")
    lora_val = False
    metricsL,_,_,_,_ = experiment(best_model, trans_ds, le, eposd, bsrt, f"{n}-LoRA", True)
    print("\nRISULTATI\n")
    print(n,"-->", mas)
    print(n,"-LoRA", "-->", metricsL["eval_f1-score"])
    if metricsL["eval_f1-score"] > mas:
        print(f"Il miglior modello è il 'Full-Fine-Tuned + LoRa'\n")
        lora_val = True

    # TRAINING FINALE
    _, best_model = build_model()
    time.sleep(5)
    _,_,_,_,trainer = experiment(best_model, trans_ds, le, eposd, bsrt, "MEGLIO_MODELLO", lora_val)

    # SALVATAGGIO
    best_model.save_pretrained("./moneta")
    feature_extractor.save_pretrained("./moneta")
    ds.save_to_disk("./moneta_dataset")
    trainer.save_model()

    print("\nVALUTAZIONE SUL MODELLO MIGLIORE")

    # Validation set
    val_metrics = trainer.evaluate(trans_ds["validation"])
    print("\nRISULTATI VALIDATION MIGLIORE:")
    trainer.log_metrics("final_validation", val_metrics)
    trainer.save_metrics("final_validation", val_metrics)

    # Test set
    test_metrics = trainer.evaluate(trans_ds["test"])
    print("\nRISULTATI TEST MIGLIORE:")
    trainer.log_metrics("final_test", test_metrics)
    trainer.save_metrics("final_test", test_metrics)

    print("\n")
    print("Validation accuracy:", val_metrics["eval_accuracy"])
    print("Validation precision:", val_metrics["eval_precision"])
    print("Validation recall:", val_metrics["eval_recall"])
    print("\n")
    print("Test accuracy:", test_metrics["eval_accuracy"])
    print("Test precision:", test_metrics["eval_precision"])
    print("Test recall:", test_metrics["eval_recall"])

    # TEST SULLA SINGOLA
    print("\n\nTest 2 modello fine-tunato sulla stessa img a caso")
    attention(best_model, img_test, "FINE TUNING")
    prediction(best_model, img_test)

if __name__ == "__main__":
    asyncio.run(main())