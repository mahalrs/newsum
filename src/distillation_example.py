import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Distill a Hugging Face summarization model")
    parser.add_argument("--teacher_model_name_or_path", type=str, required=True,
                        help="Model name or path to a pretrained teacher model")
    parser.add_argument("--student_model_name_or_path", type=str, required=True,
                        help="Model name or path to a pretrained student model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the distilled student model")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--max_epochs", type=int, default=3, help="Maximum number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size for evaluation")
    return parser.parse_args()


class DistillationTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        teacher_logits = inputs["teacher_logits"].detach()

        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        loss = loss_fn(logits.softmax(dim=-1).log(), teacher_logits.softmax(dim=-1))

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()

    # Load datasets
    train_dataset = load_dataset("text", data_files=args.train_file)["train"]
    val_dataset = load_dataset("text", data_files=args.validation_file)["train"]

    # Load teacher and student models, and tokenizer
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model_name_or_path)
    student_model = AutoModelForSeq2SeqLM.from_pretrained(args.student_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)

    # Prepare training and validation data
    def tokenize(batch):
        tokenized_input = tokenizer(batch["src_texts"], padding="max_length", truncation=True, return_tensors="pt")
        tokenized_target = tokenizer(batch["tgt_texts"], padding="max_length", truncation=True, return_tensors="pt")
        return {"src_texts": tokenized_input, "tgt_texts": tokenized_target}

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    # Generate teacher logits
    def generate_teacher_logits(batch):
        with torch.no_grad():
            input_ids = batch["src_texts"]["input_ids"]
            attention_mask = batch["src_texts"]["attention_mask"]
            teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask).logits
        return {"teacher_logits": teacher_logits}

    train_dataset = train_dataset.map(generate_teacher_logits, batched=True)
    val_dataset = val_dataset.map(generate_teacher_logits, batched=True)

    # Data collator
    data_collator = default_data_collator

    # Initialize training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.max_epochs,
        logging_dir="./logs",
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        fp16=True,
    )

    # Create trainer
    trainer = DistillationTrainer(
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train and evaluate the student model
    trainer.train()
    trainer.evaluate()

    # Save distilled student model and tokenizer
    student_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

#python distiller.py --teacher_model_name_or_path "google/pegasus-large" --student_model_name_or_path "google/pegasus-small" --output_dir "distilled_pegasus" --train_file "train_data.json" --validation_file "val_data.json" --max_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8


