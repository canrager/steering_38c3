#%%
from nnsight import LanguageModel
model_nn = LanguageModel("microsoft/phi-2", device_map="cuda", dispatch=True) # takes about 3 minutes on colab

#%%
# Get activation vector
layer_idx = 16
residual_stream = model_nn.model.layers[layer_idx] # Reinitialize the layer object

with model_nn.trace("Love"): # NOTE 1: Trace is a single forward pass, no interative, auto-regressive generation.
    happy_activation = residual_stream.output[0].save() # Confusingly layer_8.output returns a tuple, the activations we want are at idx 0

with model_nn.trace("Hate"):
    sad_activation = residual_stream.output[0].save()

act_diff = happy_activation[0, -1, :] - sad_activation[0, -1, :]
steering_factor = 10
steering_vector = steering_factor * act_diff

#%%
# Steer on prompt
prompt = "I was thinking about"

with model_nn.generate(prompt, max_new_tokens=50):
    out = residual_stream.output # Cache the current activaiton, tuple
    acts = out[0]
    acts[:, 0] += steering_vector # Modify
    residual_stream.output = (acts,) + out[1:] # Update the layer with the modified activations

    out_tokens = model_nn.generator.output.save()

out_text = model_nn.tokenizer.batch_decode(out_tokens)
print(out_text)

# %%
