// Toggle between forms
document.getElementById('detailedFormBtn').addEventListener('click', () => {
    document.getElementById('detailedForm').classList.remove('hidden');
    document.getElementById('simpleForm').classList.add('hidden');
    document.getElementById('detailedFormBtn').classList.add('bg-indigo-600', 'text-white');
    document.getElementById('detailedFormBtn').classList.remove('bg-gray-300', 'text-gray-800');
    document.getElementById('simpleFormBtn').classList.remove('bg-indigo-600', 'text-white');
    document.getElementById('simpleFormBtn').classList.add('bg-gray-300', 'text-gray-800');
});

document.getElementById('simpleFormBtn').addEventListener('click', () => {
    document.getElementById('simpleForm').classList.remove('hidden');
    document.getElementById('detailedForm').classList.add('hidden');
    document.getElementById('simpleFormBtn').classList.add('bg-indigo-600', 'text-white');
    document.getElementById('simpleFormBtn').classList.remove('bg-gray-300', 'text-gray-800');
    document.getElementById('detailedFormBtn').classList.remove('bg-indigo-600', 'text-white');
    document.getElementById('detailedFormBtn').classList.add('bg-gray-300', 'text-gray-800');
});

// Default to detailed form
document.getElementById('detailedForm').classList.remove('hidden');
document.getElementById('detailedFormBtn').classList.add('bg-indigo-600', 'text-white');
document.getElementById('simpleFormBtn').classList.add('bg-gray-300', 'text-gray-800');

document.getElementById('detailedForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSubmit(e.target);
});

document.getElementById('simpleForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    await handleSubmit(e.target);
});

async function handleSubmit(form) {
    // Show loading spinner
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('result').classList.add('hidden');
    document.getElementById('error').classList.add('hidden');

    let formData;
    if (form.id === 'detailedForm') {
        formData = {
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
    } else { // simpleForm
        const avgTransactionAmount = parseFloat(document.getElementById('avgTransactionAmount').value);
        const lastTransactionDate = new Date(document.getElementById('lastTransactionDate').value).toISOString().split('T')[0] + 'T00:00:00Z';
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
            Amount: avgTransactionAmount
        };
    }

    console.log('Sending data:', JSON.stringify(formData));

    try {
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

        if (!poorResponse.ok) {
            const errorText = await poorResponse.text();
            throw new Error(`Poor prediction failed: ${poorResponse.status} - ${errorText}`);
        }
        if (!goodResponse.ok) {
            const errorText = await goodResponse.text();
            throw new Error(`Good prediction failed: ${goodResponse.status} - ${errorText}`);
        }

        const poorData = await poorResponse.json();
        const goodData = await goodResponse.json();

        document.getElementById('loading').classList.add('hidden');
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('prediction').textContent = `Credit Score: ${poorData.prediction === 1 || goodData.prediction === 1 ? 1 : 0}`;
        document.getElementById('rfmsScore').textContent = `RFMS Score: ${poorData.rfms_score ? poorData.rfms_score.toFixed(2) : 'N/A'}`;

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
        console.error('Fetch error:', error);
    }
}