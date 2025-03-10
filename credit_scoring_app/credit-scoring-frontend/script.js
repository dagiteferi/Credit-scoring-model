document.getElementById('creditForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Show loading spinner
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

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
        TransactionStartTime: new Date(document.getElementById('transactionStartTime').value).toISOString(),
        Amount: parseFloat(document.getElementById('amount').value)
    };

    try {
        // Test both endpoints and choose the most likely prediction
        const [poorResponse, goodResponse] = await Promise.all([
            fetch('http://127.0.0.1:8000/predict/poor', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            }),
            fetch('http://127.0.0.1:8000/predict/good', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            })
        ]);

        const poorData = await poorResponse.json();
        const goodData = await goodResponse.json();

        // Hide loading spinner
        document.getElementById('loading').classList.add('hidden');

        // Display results
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('prediction').textContent = `Credit Score: ${poorData.prediction === 1 || goodData.prediction === 1 ? 1 : 0}`;
        document.getElementById('rfmsScore').textContent = `RFMS Score: ${poorData.rfms_score.toFixed(2)}`;

        // Update status indicator
        const statusIndicator = document.getElementById('statusIndicator');
        if (poorData.prediction === 1 || goodData.prediction === 1) {
            statusIndicator.className = 'w-16 h-16 mx-auto mt-4 rounded-full bg-green-500';
        } else {
            statusIndicator.className = 'w-16 h-16 mx-auto mt-4 rounded-full bg-red-500';
        }

    } catch (error) {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('error').classList.remove('hidden');
        document.getElementById('error').textContent = `Error: ${error.message}`;
    }
});