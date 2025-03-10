// Toggle between forms
document.getElementById('detailedFormBtn').addEventListener('click', () => {
    document.getElementById('detailedForm').classList.remove('hidden');
    document.getElementById('simpleForm').classList.add('hidden');
    document.getElementById('detailedFormBtn').classList.add('active');
    document.getElementById('simpleFormBtn').classList.remove('active');
});

document.getElementById('simpleFormBtn').addEventListener('click', () => {
    document.getElementById('simpleForm').classList.remove('hidden');
    document.getElementById('detailedForm').classList.add('hidden');
    document.getElementById('simpleFormBtn').classList.add('active');
    document.getElementById('detailedFormBtn').classList.remove('active');
});

// Default to detailed form
document.getElementById('detailedForm').classList.remove('hidden');
document.getElementById('detailedFormBtn').classList.add('active');

document.getElementById('detailedForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSubmit(e.target);
});

document.getElementById('simpleForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSubmit(e.target);
});

async function handleSubmit(form) {
    console.log("Starting handleSubmit for form:", form.id);

    // Show loading spinner
    console.log("Showing loading spinner...");
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    let formData;
    if (form.id === 'detailedForm') {
        formData = {
            TransactionId: parseInt(document.getElementById('transactionId').value) || 0,
            BatchId: parseInt(document.getElementById('batchId').value) || 0,
            AccountId: parseInt(document.getElementById('accountId').value) || 0,
            SubscriptionId: parseInt(document.getElementById('subscriptionId').value) || 0,
            CustomerId: parseInt(document.getElementById('customerId').value) || 0,
            CurrencyCode: document.getElementById('currencyCode').value || 'UGX',
            CountryCode: document.getElementById('countryCode').value || '256',
            ProductId: parseInt(document.getElementById('productId').value) || 0,
            ChannelId: parseInt(document.getElementById('channelId').value) || 2,
            TransactionStartTime: new Date(document.getElementById('transactionStartTime').value).toISOString() || new Date().toISOString(),
            Amount: parseFloat(document.getElementById('amount').value) || 0,
            FraudResult: parseInt(document.getElementById('fraudResult').value) || 0
        };
    } else { // simpleForm
        const avgTransactionAmount = parseFloat(document.getElementById('avgTransactionAmount').value) || 0;
        const lastTransactionDate = new Date(document.getElementById('lastTransactionDate').value).toISOString().split('T')[0] + 'T00:00:00Z' || new Date().toISOString();
        formData = {
            TransactionId: 1,
            BatchId: 1,
            AccountId: 1,
            SubscriptionId: 1,
            CustomerId: 256,
            CurrencyCode: 'UGX',
            CountryCode: '256',
            ProductId: 0,
            ChannelId: 2,
            TransactionStartTime: lastTransactionDate,
            Amount: avgTransactionAmount,
            FraudResult: parseInt(document.getElementById('simpleFraudResult').value) || 0
        };
    }

    console.log('Form Data:', formData);
    console.log('Sending data as JSON:', JSON.stringify(formData));

    try {
        console.log("Sending requests to /predict/poor and /predict/good...");
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

        console.log("Poor response status:", poorResponse.status);
        console.log("Good response status:", goodResponse.status);

        if (!poorResponse.ok) {
            const errorText = await poorResponse.text();
            throw new Error(`Poor prediction failed: ${poorResponse.status} - ${errorText}`);
        }
        if (!goodResponse.ok) {
            const errorText = await goodResponse.text();
            throw new Error(`Good prediction failed: ${poorResponse.status} - ${errorText}`);
        }

        const poorData = await poorResponse.json();
        const goodData = await goodResponse.json();

        console.log("Poor prediction data:", poorData);
        console.log("Good prediction data:", goodData);

        const finalCreditScore = poorData.prediction === 1 || goodData.prediction === 1 ? 1 : 0;

        console.log("Updating results section with credit score:", finalCreditScore);
        document.getElementById('loading').classList.add('hidden');
        const resultElement = document.getElementById('result');
        if (resultElement) {
            console.log("Result element found, removing hidden class and setting display...");
            resultElement.classList.remove('hidden');
            resultElement.style.display = 'block'; // Force display
            resultElement.style.opacity = '1'; // Override transition opacity
            resultElement.style.transform = 'translateY(0)'; // Override transition transform
            document.getElementById('creditScore').textContent = finalCreditScore;
            document.getElementById('rfmsScore').textContent = `RFMS Score: ${poorData.rfms_score ? poorData.rfms_score.toFixed(2) : 'N/A'}`;
            document.getElementById('riskStatus').textContent = finalCreditScore === 1 ? 'Low Risk' : 'High Risk';
            document.getElementById('recommendation').textContent = finalCreditScore === 1 ? 'Recommend Approval' : 'Further Verification Required';
            const statusIndicator = document.getElementById('statusIndicator');
            if (statusIndicator) {
                statusIndicator.style.backgroundColor = finalCreditScore === 1 ? 'var(--success)' : 'var(--danger)';
            } else {
                console.error("Status indicator element not found!");
            }
            console.log("Results section updated.");
        } else {
            console.error("Result element not found in DOM!");
        }
    } catch (error) {
        console.error("Error in handleSubmit:", error);
        document.getElementById('loading').classList.add('hidden');
        const errorElement = document.getElementById('error');
        if (errorElement) {
            errorElement.classList.remove('hidden');
            errorElement.textContent = `Error: ${error.message}`;
        } else {
            console.error("Error element not found!");
        }
        console.log("Error displayed:", error.message);
    }
}