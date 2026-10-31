# Strata on Kaggle — SC117 Abliterated Qwen3.8-Flash-Next

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/seczoid/devzonex-strata-kaggle-abliterated/blob/main/notebooks/Strata_Kaggle_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Kaggle%20GPU-20BEFF)
![Status](https://img.shields.io/badge/status-experimental-orange)

**Import the notebook, configure Kaggle credentials and GPU, then run the cells in order.** This fork adapts the DevZoneX Strata Kaggle workflow to [SC117's abliterated Qwen3.8-Flash-Next GSQ-RCO GGUF](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF).

It retains the original workflow's Kaggle Secrets integration, private model/runtime datasets, Strata runtime cache, authenticated OpenAI-compatible API, Cloudflare Quick Tunnel, end-to-end API verification, and OpenCode configuration output. The default model is SC117 **Q2_0**, not the upstream **gyro / gyro-s** preset.

## One-click setup

1. Click **Open in Kaggle** above.
2. In Kaggle, enable **Internet** and a **GPU accelerator**.
3. Add a Kaggle Secret named **KAGGLE_API_TOKEN** with your Kaggle API token.
4. In the notebook's first configuration cell, set **KAGGLE_USERNAME** to your Kaggle username.
5. Read the model card and its license/terms. Set **I_CONFIRM_MODEL_LICENSE_AND_PRIVATE_STORAGE = True** only if you agree to the terms and want to upload the weights to your own private Kaggle Dataset.
6. Run notebook cells from top to bottom.

The first run is a substantial setup: the selected GGUF tier is tens of gigabytes, a model-specific native pack is built, and the compatible Strata engine may need to be installed. Do not interrupt the notebook while a download, pack build, or Kaggle dataset upload is in progress.

## Model and quantization

The notebook defaults to:

- **Repository:** SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF
- **Tier:** Q2_0
- **Context:** 32k
- **MTP:** SC117's published Strata runtime

The configuration cell also allows **IQ2_XS**, **IQ3_XXS**, or **IQ3_S**. Choose a tier that exists in the Hugging Face repository and that fits the Kaggle session's RAM, VRAM, and disk budget. The GPU assigned to a Kaggle session is not guaranteed to be the same every time.

### Why the custom pack matters

Do **not** reuse the original Strata Gyro-S or official GSQ-RCO native pack. SC117's tensor offsets differ from the original model. The notebook builds a separate pack from the selected SC117 shard using Strata's **tools/iq_pack.py**, then configures the server with:

- **--pack:** the newly built packs/sc117_<tier> directory
- **--native:** SC117 shard 1
- **--ple-gguf:** SC117 shard 2
- **tokenizer:** the new pack's tokenizer directory
- **--mtp:** SC117's strata/rt runtime

See the model author's [Strata-specific instructions](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF/blob/main/strata/README.md).

**MTP caveat:** the published MTP draft head was extracted from the stock Qwen3.8-Flash-Next checkpoint and is not itself abliterated. The main model weights are the abliterated ones.

## Persistence across Kaggle sessions

The notebook uses two separate **private Kaggle Datasets**:

- **strata-model-sc117-q2-0** (the suffix changes with the selected tier): model shards, native pack, tokenizer, and MTP files.
- **strata-runtime-cache-sc117**: Strata terminal and installed engine components, plus Cloudflare's tunnel binary when available.

On the first run, the notebook downloads and prepares the model and then offers to upload the caches. Model-cache upload is deliberately disabled until you set the explicit confirmation flag to **True**. The runtime cache is uploaded separately.

On later runs, attach both datasets using Kaggle's **Add Input** control before running the notebook. The notebook restores them automatically. **MODEL_RESTORE_MODE = "symlink"** avoids an extra local copy; change it to **"copy"** if reading model weights from the Kaggle input mount is too slow and your session has enough disk space.

**Persistence has a limit:** Kaggle still stages attached dataset files into each new runtime. The cache avoids downloading the weights from Hugging Face and rebuilding the pack, but it does not eliminate Kaggle's dataset staging or restore time.

If you change the model tier, attach the matching model-cache dataset printed by the notebook. Do not rename a cache dataset to make one tier appear to be another.

## API and OpenCode

After startup, the notebook:
- waits for the model's local health check;
- starts a temporary Cloudflare Quick Tunnel;
- verifies unauthenticated API access is rejected;
- runs a short authenticated generation test; and
- prints the OpenCode provider snippet.

Writable model preparation data, server config, and API key files live under `/kaggle/working`; attached Kaggle Dataset inputs are read-only. The API key is generated locally and stored in a file with restricted permissions. The notebook does not print the key automatically. Only display it in a private notebook session when you are ready to configure OpenCode. The Quick Tunnel URL is temporary and changes between sessions; it is not a permanent hosting service.

Do not publish notebook outputs that contain your API key or other credentials.

## Troubleshooting

- **No GPU detected:** enable a GPU accelerator in Kaggle settings and rerun the helper-loading cell.
- **Hugging Face download interrupted:** rerun the model-preparation cell; the Hugging Face local download cache is reused where possible.
- **Pack build fails:** check that both shard files finished downloading and that the notebook has Internet enabled. Do not substitute a pack from another model.
- **Native expert mismatch:** remove only the incomplete **packs/sc117_<tier>** directory from the current session and rerun preparation. Never point the configuration at the upstream Gyro-S pack.
- **Runtime dataset missing:** the notebook can install Strata again, but a missing runtime cache may require downloading the compatible engine component.
- **Model-cache upload skipped:** review the model card/license and set **I_CONFIRM_MODEL_LICENSE_AND_PRIVATE_STORAGE = True** only if you explicitly want to store the model in a private Kaggle Dataset.
- **Kaggle session reset:** attach both private datasets again. Kaggle session lifetime and accelerator quotas are controlled by Kaggle.

## Project structure

- **notebooks/Strata_Kaggle_Launcher.ipynb** — Kaggle Quick Link notebook.
- **scripts/strata_kaggle.py** — shared Kaggle auth, cache, tunnel, API verification, and OpenCode helpers.
- **scripts/sc117_model.py** — SC117 download, native-pack build, engine preflight, and custom server configuration.

## Credits and license

This is a community fork of [iAmAjayTeli/devzonex-strata-kaggle](https://github.com/iAmAjayTeli/devzonex-strata-kaggle). It depends on [Niko1221/Strata](https://github.com/Niko1221/Strata) and the model files published by [SC117](https://huggingface.co/SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF). Respect the upstream repository licenses and the Hugging Face model terms. The repository's MIT license applies to this fork's own code where permitted by the upstream license.
