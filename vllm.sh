#set up wizard_coder service
python3 -m vllm.entrypoints.openai.api_server --model WizardLMTeam/WizardCoder-Python-13B-V1.0 --port 8082  --dtype auto --api-key token-abc123