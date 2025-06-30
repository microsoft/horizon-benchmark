from recbole.quick_start import load_data_and_model
from recbole.utils import init_seed, get_trainer
import sys

# Define the path to your checkpoint
checkpoint_path = sys.argv[1]

# Load the model, dataset, and dataloaders from the checkpoint
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=checkpoint_path
)

# Set evaluation arguments properly for both validation and test
config["eval_args"] = {
    'split': {'RS': [0.8, 0.1, 0.1]}, 
    'order': 'TO',
    'group_by': 'user', 
    'mode': {'valid': 'full', 'test': 'full'}
}

config["eval_args"] = {
        'split': {'LS': 'valid_and_test'}, 
        'order': 'TO',
        'group_by': 'user', 
        'mode': {'valid': 'full', 'test': 'full'}
    }

config["TIME_FIELD"] = "timestamp"
config["load_col"] = {'inter': ['user_id', 'item_id','timestamp'], 'item': ['item_id']}


# Move model to device
model.to(config['device'])

# Get trainer
trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

# Evaluate on validation set
print("Validation Results:")
valid_result = trainer.evaluate(valid_data)
print(valid_result)

# Evaluate on test set
print("\nTest Results:")
test_result = trainer.evaluate(test_data)
print(test_result)