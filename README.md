# Clinical Trial Matching API

An advanced AI-powered system for matching patients with suitable clinical trials using a sophisticated multi-step pipeline that combines traditional information retrieval techniques with state-of-the-art language models.

## 🌟 Features

- **Hybrid Retrieval System**: Combines multiple retrieval methods like BM-25 and MedCPT for optimal keywords plus semantic trial retrieval
- **AI-Powered Matching**: Utilizes GPT models for intelligent trial-patient compatibility analysis
- **Comprehensive Pipeline**: Multi-step process including keyword generation, matching, aggregation, and ranking
- **Scalable Architecture**: Designed to handle multiple trials and complex patient data
- **Detailed Logging**: Comprehensive logging system for debugging and monitoring
- **Error Resilient**: Robust error handling at each pipeline stage

## 🏗️ System Architecture

The system implements a pipeline architecture with the following components:

1. **Keyword Generation**: Extracts relevant medical conditions from patient notes
2. **Hybrid Retrieval**: Identifies potentially relevant clinical trials
3. **Matching Process**: Detailed compatibility analysis between patients and trials
4. **Aggregation**: Combines multiple factors for comprehensive assessment
5. **Ranking**: Produces final ordered list of recommended trials

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clinical-trial-matching-api.git
cd clinical-trial-matching-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Download the dataset from clinicaltrials.gov in jsonl format and paste in the Dataset folder.
   
## 💻 Usage

### Basic Usage

```python
from app.tasks import ClinicalTrialRetrievalPipeline

# Initialize the pipeline
pipeline = ClinicalTrialRetrievalPipeline(model="gpt-4o-mini")

# Process a patient
result = pipeline.execute_pipeline(
    patient_id="patient123",
    patient_text="Patient medical history and conditions...",
    N=20  # Number of trials to retrieve
)
```

### API Response Format

```json
{
    "patient_id": "patient123",
    "nctids": ["NCT123", "NCT456", ...],
    "rank_scores": [
        {"trial_id": "NCT123", "score": 0.95},
        {"trial_id": "NCT456", "score": 0.85}
    ],
    "final_scores": [...],
    "matching_results": {...},
    "aggregation_results": {...}
}
```

## 🔧 Configuration

The system can be configured through various parameters:

- Model selection for AI components
- Number of trials to retrieve
- Logging levels and formats
- Matching thresholds and parameters

## 📊 Pipeline Components

### 1. Keyword Generation
- Extracts relevant medical conditions from patient notes
- Generates search keywords for trial matching

### 2. Hybrid Retrieval
- Combines multiple retrieval methods
- Returns top-N most relevant clinical trials

### 3. Matching Process
- Detailed analysis of patient-trial compatibility
- Uses AI models for intelligent matching

### 4. Aggregation
- Combines multiple matching factors
- Produces comprehensive trial assessments

### 5. Ranking
- Final scoring and ordering of trials
- Produces ranked recommendations

## 🛠️ Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for their valuable tools and libraries


## ⚠️ Disclaimer

This software is intended for research and development purposes. Always consult with healthcare professionals for medical decisions. 
