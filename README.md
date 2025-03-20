# GSC BigQuery Analyzer Deployment Guide

## Local Development

1. **Set Up Python Environment**

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

2. **Run the App Locally**

```bash
streamlit run app.py
```

## GCP Service Account Setup

1. **Create a Service Account**

- Go to the [Google Cloud Console](https://console.cloud.google.com/)
- Navigate to "IAM & Admin" > "Service Accounts"
- Click "Create Service Account"
- Name it (e.g., "gsc-bigquery-analyzer")
- Add the following roles:
  - BigQuery Data Viewer
  - BigQuery Job User
  - BigQuery User
- Click "Done"

2. **Create and Download Service Account Key**

- Find your service account in the list
- Click the three dots menu > "Manage keys"
- Click "Add Key" > "Create new key"
- Choose JSON format
- Download the key file

3. **Secure the Key**

- Never commit the key to version control
- Set it as an environment variable or use a secrets manager

## Docker Deployment

1. **Build the Docker Image**

```bash
docker build -t gsc-bigquery-analyzer .
```

2. **Run the Container**

```bash
docker run -p 8501:8501 -v /path/to/service-account-key.json:/app/key.json -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json gsc-bigquery-analyzer
```

## Streamlit Cloud Deployment

1. **Create a GitHub Repository**
   - Push your code to a GitHub repository
   - Make sure to exclude any service account keys!

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Connect your GitHub account
   - Select your repository
   - Choose the main file (app.py)
   - Add the following secret:
     - Name: `GOOGLE_APPLICATION_CREDENTIALS_JSON`
     - Value: Copy the entire content of your service account JSON key file

3. **Add this to the top of your app.py**

```python
# For Streamlit Cloud deployment
if 'GOOGLE_APPLICATION_CREDENTIALS_JSON' in os.environ:
    import json
    service_account_info = json.loads(os.environ['GOOGLE_APPLICATION_CREDENTIALS_JSON'])
    credentials = service_account.Credentials.from_service_account_info(service_account_info)
    st.session_state['bq_client'] = bigquery.Client(credentials=credentials)
```

## Google Cloud Run Deployment

1. **Push the Docker Image to Google Container Registry**

```bash
# Tag the image
docker tag gsc-bigquery-analyzer gcr.io/YOUR_PROJECT_ID/gsc-bigquery-analyzer

# Push to GCR
gcloud auth configure-docker
docker push gcr.io/YOUR_PROJECT_ID/gsc-bigquery-analyzer
```

2. **Deploy to Cloud Run**

```bash
gcloud run deploy gsc-bigquery-analyzer \
  --image gcr.io/YOUR_PROJECT_ID/gsc-bigquery-analyzer \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi
```

3. **Set Up Service Account for Cloud Run**

```bash
# Create a service account for Cloud Run
gcloud iam service-accounts create gsc-bigquery-runner

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:gsc-bigquery-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.user"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:gsc-bigquery-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:gsc-bigquery-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"

# Update the Cloud Run service to use this service account
gcloud run services update gsc-bigquery-analyzer \
  --service-account=gsc-bigquery-runner@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

## Scheduled Data Import Setup

1. **Create a BigQuery Scheduled Query**

- Go to BigQuery Console
- Click "Create Query"
- Paste the query template from the app
- Click "Schedule" button
- Set up the schedule according to your needs
- Set the destination table
- Enable the schedule

2. **Monitor the Scheduled Query**

- Check the "Scheduled Queries" section in BigQuery
- Review logs and errors
- Adjust the schedule as needed

## Maintenance

- Regularly update dependencies
- Monitor logs for errors
- Check for performance bottlenecks
- Review and optimize BigQuery costs

## Troubleshooting

- **Authentication Issues**: Verify service account permissions and key validity
- **Data Not Loading**: Check BigQuery table structure and permissions
- **Performance Problems**: Consider optimizing BigQuery queries or increasing memory allocation
- **Visualization Errors**: Verify data types and ensure no nulls in critical columns
