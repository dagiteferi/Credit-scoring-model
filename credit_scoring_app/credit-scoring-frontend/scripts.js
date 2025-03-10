const BASE_URL = window.location.hostname.includes("localhost")
    ? "http://127.0.0.1:8000"
    : "https://credit-scoring-model-ai.onrender.com";
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

function generateExplanation(formData, poorData, goodData, finalCreditScore) {
    const currentDate = new Date(2025, 2, 10); // March 10, 2025 (month is 0-based)
    const transactionDate = new Date(formData.TransactionStartTime);
    const daysDifference = Math.floor((currentDate - transactionDate) / (1000 * 60 * 60 * 24));
    const yearsDifference = Math.floor(daysDifference / 365);
    const rfmsScore = poorData.rfms_score || goodData.rfms_score;

    let explanation = [];
    if (finalCreditScore === 1) { // Low risk
        explanation.push("This transaction is low risk with an RFMS Score of " + rfmsScore.toFixed(2) + ".");
        if (formData.FraudResult === 0) {
            explanation.push("No fraud (FraudResult: 0) supports this.");
        }
        if (daysDifference > 365) {
            explanation.push("The " + yearsDifference + "-year-old transaction is offset by strong customer activity.");
        }
        if (formData.Amount < 1.0) {
            explanation.push("The low amount (" + formData.Amount + " " + formData.CurrencyCode + ") is balanced by overall engagement.");
        }
    } else { // High risk
        explanation.push("This transaction is high risk with an RFMS Score of " + rfmsScore.toFixed(2) + ".");
        if (formData.FraudResult === 1) {
            explanation.push("Fraud (FraudResult: 1) increases the risk.");
        }
        if (daysDifference > 365) {
            explanation.push("The " + yearsDifference + "-year-old transaction indicates low recency.");
        }
        if (formData.Amount < 1.0) {
            explanation.push("The low amount (" + formData.Amount + " " + formData.CurrencyCode + ") suggests limited activity.");
        }
    }
    return explanation.join(" ");
}

function openDetailedView(formData, poorData, goodData, finalCreditScore) {
    const currentDate = new Date(2025, 2, 10); // March 10, 2025
    const transactionDate = new Date(formData.TransactionStartTime);
    const daysDifference = Math.floor((currentDate - transactionDate) / (1000 * 60 * 60 * 24));
    const yearsDifference = Math.floor(daysDifference / 365);
    const rfmsScore = poorData.rfms_score || goodData.rfms_score;

    let detailedContent = `
        <html>
        <head>
            <title>Detailed Risk Analysis</title>
            <style>
                body { 
                    font-family: 'Arial', sans-serif; 
                    margin: 20px; 
                    background: #f8f9fa; 
                    color: #202124; 
                    line-height: 1.6; 
                }
                h2 { 
                    color: #1a73e8; 
                    font-size: 1.8em; 
                    border-bottom: 2px solid #1a73e8; 
                    padding-bottom: 5px; 
                    margin-bottom: 20px; 
                }
                p { 
                    font-size: 1em; 
                    margin: 10px 0; 
                }
                ul { 
                    list-style-type: disc; 
                    padding-left: 30px; 
                    margin: 10px 0; 
                }
                li { 
                    margin: 5px 0; 
                    font-size: 0.95em; 
                }
                .graph-container { 
                    margin: 30px 0; 
                    padding: 15px; 
                    background: #ffffff; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); 
                }
                canvas { 
                    max-width: 100%; 
                    border-radius: 5px; 
                }
            </style>
        </head>
        <body>
            <h2>Detailed Risk Analysis</h2>
            <p><strong>Transaction Details:</strong></p>
            <ul>
                <li>Customer ID: ${formData.CustomerId}</li>
                <li>Transaction Date: ${transactionDate.toISOString().split('T')[0]}</li>
                <li>Amount: ${formData.Amount} ${formData.CurrencyCode}</li>
                <li>Fraud Result: ${formData.FraudResult === 0 ? 'Not Fraudulent' : 'Fraudulent'}</li>
                <li>Channel: ${formData.ChannelId === 2 ? 'Online' : formData.ChannelId === 3 ? 'Mobile App' : 'POS Terminal'}</li>
                <li>RFMS Score: ${rfmsScore.toFixed(2)}</li>
                <li>Credit Score: ${finalCreditScore} (${finalCreditScore === 1 ? 'Low Risk' : 'High Risk'})</li>
            </ul>
            <p><strong>How the Label is Determined:</strong></p>
            <p>The 'Label' (0 or 1) is based on the RFMS score, calculated as:</p>
            <ul>
                <li><strong>Recency:</strong> Days since transaction (${daysDifference} days, or ${yearsDifference} years). Score contribution: 1 / (Recency + 1) * 0.4 (lower Recency boosts score).</li>
                <li><strong>Frequency:</strong> Number of transactions per customer. Score contribution: Transaction_Count * 0.3 (higher counts increase score).</li>
                <li><strong>Monetary:</strong> Total transaction amount per customer. Score contribution: Total_Transaction_Amount * 0.3 (higher totals improve score).</li>
            </ul>
            <p>Formula: RFMS_score = (1 / (Recency + 1) * 0.4) + (Transaction_Count * 0.3) + (Total_Transaction_Amount * 0.3)</p>
            <p>The Label is 1 if RFMS_score exceeds the dataset median, 0 otherwise. WOE adjusts bins of RFMS scores based on historical risk patterns.</p>
            <p><strong>How the Result is Calculated:</strong></p>
            <p>The credit score (0 or 1) comes from API predictions:</p>
            <ul>
                <li>/predict/poor and /predict/good return a prediction (0 or 1).</li>
                <li>Final Score = 1 if either prediction is 1, 0 if both are 0.</li>
                <li>For your data, an RFMS score of ${rfmsScore.toFixed(2)} and no fraud led to the current prediction.</li>
            </ul>
            <p><strong>Why This Result?</strong></p>
            <p>${finalCreditScore === 1 ? 
                "The score of 1 (Low Risk) reflects a high RFMS score (" + rfmsScore.toFixed(2) + "), suggesting strong engagement. The old transaction and low amount are offset by frequent activity and no fraud (FraudResult: 0)." : 
                "The score of 0 (High Risk) reflects a low RFMS score (" + rfmsScore.toFixed(2) + "), possibly due to infrequent transactions, an old date, or limited activity."}</p>
            <div class="graph-container">
                <canvas id="rfmScatter"></canvas>
            </div>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
                const ctx = document.getElementById('rfmScatter').getContext('2d');
                new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [{
                            label: 'Transaction Point',
                            data: [{x: ${daysDifference}, y: ${formData.Amount}, r: 15}],
                            backgroundColor: 'rgba(26, 115, 232, 0.7)', // Vibrant blue with slight transparency
                            borderColor: '#1a73e8',
                            borderWidth: 2,
                            pointHoverRadius: 20,
                            pointHoverBackgroundColor: '#1a73e8',
                            pointHoverBorderColor: '#ffffff',
                            pointHoverBorderWidth: 3
                        }]
                    },
                    options: {
                        plugins: {
                            title: {
                                display: true,
                                text: 'Transaction Recency vs. Amount',
                                font: { size: 16, weight: 'bold' },
                                color: '#202124',
                                padding: { top: 10, bottom: 20 }
                            },
                            legend: {
                                position: 'top',
                                labels: {
                                    font: { size: 12 },
                                    color: '#202124'
                                }
                            },
                            tooltip: {
                                enabled: true,
                                callbacks: {
                                    label: function(context) {
                                        let label = context.dataset.label || '';
                                        if (label) {
                                            label += ': ';
                                        }
                                        label += \`Recency: \${context.parsed.x} days, Amount: \${context.parsed.y} UGX\`;
                                        return label;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Recency (Days)',
                                    font: { size: 14 },
                                    color: '#202124'
                                },
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                ticks: {
                                    color: '#202124',
                                    stepSize: 100 // Adjust for better readability
                                },
                                min: ${daysDifference - 50}, // Center the point
                                max: ${daysDifference + 50}
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Amount (UGX)',
                                    font: { size: 14 },
                                    color: '#202124'
                                },
                                grid: {
                                    color: 'rgba(0, 0, 0, 0.05)'
                                },
                                ticks: {
                                    color: '#202124',
                                    stepSize: 0.01 // Adjust for small amounts
                                },
                                min: 0,
                                max: 0.06 // Adjust to fit the single point
                            }
                        },
                        responsive: true,
                        maintainAspectRatio: false
                    }
                });
            </script>
        </body>
        </html>
    `;

    const newTab = window.open('', '_blank');
    newTab.document.write(detailedContent);
    newTab.document.close();
}

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
            resultElement.style.display = 'block';
            resultElement.style.opacity = '1';
            resultElement.style.transform = 'translateY(0)';
            document.getElementById('creditScore').textContent = finalCreditScore;
            document.getElementById('rfmsScore').textContent = `RFMS Score: ${poorData.rfms_score ? poorData.rfms_score.toFixed(2) : 'N/A'}`;
            document.getElementById('riskStatus').textContent = finalCreditScore === 1 ? 'Low Risk' : 'High Risk';
            document.getElementById('recommendation').textContent = finalCreditScore === 1 ? 'Recommend Approval' : 'Further Verification Required';
            document.getElementById('explanation').textContent = generateExplanation(formData, poorData, goodData, finalCreditScore);
            const seeMoreBtn = document.getElementById('seeMoreBtn');
            seeMoreBtn.onclick = () => openDetailedView(formData, poorData, goodData, finalCreditScore);
            const statusIndicator = document.getElementById('statusIndicator');
            if (statusIndicator) {
                console.log("Setting status indicator color...");
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