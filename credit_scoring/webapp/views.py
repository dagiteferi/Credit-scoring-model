from django.shortcuts import render
from django.http import JsonResponse
import requests
import json

def home_view(request):
    return render(request, 'home.html')

def get_score(request):
    if request.method == 'POST':
        data = {
            'ProviderId': request.POST.get('providerId'),
            'ProductCategory': request.POST.get('productCategory'),
            'Amount': request.POST.get('amount'),
            'Value': request.POST.get('value'),
            'PricingStrategy': request.POST.get('pricingStrategy'),
            'FraudResult': request.POST.get('fraudResult'),
            'Total_Transaction_Amount': request.POST.get('totalTransactionAmount'),
            'Average_Transaction_Amount': request.POST.get('averageTransactionAmount'),
            'Transaction_Count': request.POST.get('transactionCount'),
            'Std_Transaction_Amount': request.POST.get('stdTransactionAmount'),
            'Transaction_Hour': request.POST.get('transactionHour'),
            'Transaction_Day': request.POST.get('transactionDay'),
            'Transaction_Month': request.POST.get('transactionMonth'),
            'Transaction_Year': request.POST.get('transactionYear'),
            'CurrencyCode_WOE': request.POST.get('currencyCodeWOE'),
            'ProviderId_WOE': request.POST.get('providerIdWOE'),
            'ProductId_WOE': request.POST.get('productIdWOE'),
            'ProductCategory_WOE': request.POST.get('productCategoryWOE'),
            'Recency': request.POST.get('recency'),
            'RFMS_score': request.POST.get('rfmsScore'),
            'ProductId_1': request.POST.get('productId1') == 'on',
            'ProductId_2': request.POST.get('productId2') == 'on',
            'ProductId_3': request.POST.get('productId3') == 'on',
            'ProductId_4': request.POST.get('productId4') == 'on',
            'ProductId_5': request.POST.get('productId5') == 'on',
            'ProductId_6': request.POST.get('productId6') == 'on',
            'ProductId_7': request.POST.get('productId7') == 'on',
            'ProductId_8': request.POST.get('productId8') == 'on',
            'ProductId_9': request.POST.get('productId9') == 'on',
            'ProductId_10': request.POST.get('productId10') == 'on',
            'ProductId_11': request.POST.get('productId11') == 'on',
            'ProductId_12': request.POST.get('productId12') == 'on',
            'ProductId_13': request.POST.get('productId13') == 'on',
            'ProductId_14': request.POST.get('productId14') == 'on',
            'ProductId_15': request.POST.get('productId15') == 'on',
            'ProductId_16': request.POST.get('productId16') == 'on',
            'ProductId_17': request.POST.get('productId17') == 'on',
            'ProductId_18': request.POST.get('productId18') == 'on',
            'ProductId_19': request.POST.get('productId19') == 'on',
            'ProductId_20': request.POST.get('productId20') == 'on',
            'ProductId_21': request.POST.get('productId21') == 'on',
            'ProductId_22': request.POST.get('productId22') == 'on',
            'ChannelId_ChannelId_2': request.POST.get('channelId2') == 'on',
            'ChannelId_ChannelId_3': request.POST.get('channelId3') == 'on',
            'ChannelId_ChannelId_5': request.POST.get('channelId5') == 'on',
            'TransactionHour': request.POST.get('transactionHour'),
            'TransactionDay': request.POST.get('transactionDay'),
            'TransactionMonth': request.POST.get('transactionMonth'),
            'TransactionWeekday': request.POST.get('transactionWeekday'),
            'ProductId': request.POST.get('productId')
        }

        response = requests.post('http://127.0.0.1:8000/api/predict/', headers={'Content-Type': 'application/json'}, data=json.dumps(data))
        prediction = response.json().get('prediction')
        return JsonResponse({'prediction': prediction})

    return JsonResponse({'error': 'Invalid request'}, status=400)
