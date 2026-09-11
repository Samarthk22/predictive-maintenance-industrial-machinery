# IBM Cloud Deployment

Use IBM watsonx.ai Runtime (formerly Watson Machine Learning) for the production ML deployment.

## Recommended IBM services
- IBM Cloud Object Storage
- IBM watsonx.ai Studio / Project
- IBM watsonx.ai Runtime
- Optional IBM Watson OpenScale

## Steps
1. Create/open an IBM Cloud project.
2. Add Cloud Object Storage.
3. Upload `predictive_maintenance.csv`.
4. Run `ml/train.py` in an IBM notebook or import the trained model.
5. Save the model to the project.
6. Promote it to a deployment space.
7. Create an **Online** deployment.
8. Wait for status `Deployed`.
9. Open the deployment API reference and copy the scoring endpoint.
10. Create an IBM Cloud API key.
11. Put the endpoint and key in `.env`:

```env
IBM_API_KEY=YOUR_KEY
IBM_SCORING_URL=YOUR_SCORING_ENDPOINT
USE_IBM=true
BACKEND_URL=http://localhost:8000
```

12. Start FastAPI and Streamlit.

IBM's current deployment documentation:
https://www.ibm.com/docs/en/watsonx/saas?topic=assets-deploying-machine-learning
