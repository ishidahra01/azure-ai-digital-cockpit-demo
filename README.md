# azure-ai-digital-cockpit-demo

This repository provides a **demonstration application of an AI agent for a vehicle digital cockpit**, built using a Small Language Model called **Phi3.5**. By leveraging user intent and profile data, this AI agent can recommend the ideal in-car environment and relevant vehicle APIs or actions, serving as a sample framework for developing your own agent in a digital cockpit setting.

## Features

- **SLM Phi3.5** based AI agent
- Fine-tuning Phi3.5 to enable **function calling**
- Multiple agents composed **without specialized frameworks** (illustrating the agent concept)
- **Streamlit**-based web demonstration application

## Prerequisites

Before you begin, ensure that the following requirements are met:

- Access to **Azure OpenAI Service**
- An **Azure Machine Learning Workspace**  
  - Sufficient quota to create and start an **NVIDIA A100 GPU** node (e.g., Standard_NC24ads_A100_v4) in a compute cluster
- Access to **Azure AI Foundry**
- Phi3.5 **already deployed** in **Model Catalog**
- Add `Storage Blob Data Contributor` to your Azure account

## Usage

1. **Clone or download** this repository:
   ```bash
   git clone https://github.com/ishidahra01/azure-ai-digital-cockpit-demo
   cd azure-ai-digital-cockpit-demo
   ```

2. **Set up environment variables** in your `.env` file (e.g., API keys, resource names).

3. **Fine-tune the Phi3.5 model** using Azure Machine Learning:
   1. Configure a compute instance in Azure ML with the necessary Conda environment and Python packages:
      ```bash
      conda activate azureml_py310_sdkv2
      pip install -r requirements.txt
	  pip install flash-attn --no-build-isolation
      ```
   2. Use `SFT_Synthetic_Data.ipynb` to **generate synthetic training data** for fine-tuning.
   3. Use `SFT_FN_Calling_InCar_AzureML.ipynb` to **fine-tune** the model on Azure ML.
   4. Use `SFT_FN_Calling_InCar_AzureML_Deploy.ipynb` to **deploy** the fine-tuned model as an endpoint in Azure ML.

4. **Test the Agent Workflow**:
   - Refer to `src/02_agent-workflow/02_agent_get_started.ipynb` for the initial workflow code.
   - This notebook can be run locally (e.g., in VSCode) or on Azure ML.
   - Install the necessary Python packages using the `requirements.txt` file in the repository root.

5. **Run the Agent Workflow Web Demo**:
   - Launch the Streamlit app by running the following command:
     ```bash
     streamlit run src/02_agent-workflow/02_demo_app.py
     ```

6. **[Option]Run the Agent Workflow with Autogen**:
   - Refer to `src/02_agent-workflow/02_agent_get_started.ipynb` for the initial workflow code.
   - This notebook can be run locally (e.g., in VSCode) or on Azure ML.

## Contributing

Contributions are welcome! Whether it’s reporting bugs, suggesting new features, or submitting pull requests, please feel free to get involved.

1. Open an Issue to discuss any bugs or proposed changes.
2. Fork the repository, implement your changes, and open a pull request.
3. Await review and merge into the main branch.

## License

This project is open-sourced under the [MIT License](LICENSE). For more details, please see the [LICENSE](LICENSE) file.