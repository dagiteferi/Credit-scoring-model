document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = {
        TransactionId: parseInt(document.getElementById('transactionId').value),
        BatchId: parseInt(document.getElementById('batchId').value),
        AccountId: parseInt(document.getElementById('accountId').value),
        SubscriptionId: parseInt(document.getElementById('subscriptionId').value),
        CustomerId: parseInt(document.getElementById('customerId').value),
        CurrencyCode: document.getElementById('currencyCode').value,
        CountryCode: document.getElementById('countryCode').value,
        ProductId: parseInt(document.getElementById('productId').value),
        ChannelId: parseInt(document.getElementById('channelId').value),
        TransactionStartTime: document.getElementById('transactionStartTime').value,
        Amount: parseFloat(document.getElementById('amount').value)
    };

    // API endpoint (update with your deployed URL or local URL)
    const apiUrl = 'http://localhost:8000/predict'; // Change to Heroku URL after deployment, e.g., https://credit-scoring-api.herokuapp.com/predict

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        if (!response.ok) throw new Error('Network response was not ok');

        const result = await response.json();
        document.getElementById('result').innerHTML = `
            <p>Prediction: ${result.prediction} (${result.prediction === 0 ? 'Low Risk' : 'High Risk'})</p>
            <p>RFMS Score: ${result.rfms_score.toFixed(2)}</p>
            <p>Credit Score: ${result.credit_score} / 800</p>
        `;
    } catch (error) {
        document.getElementById('result').innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});