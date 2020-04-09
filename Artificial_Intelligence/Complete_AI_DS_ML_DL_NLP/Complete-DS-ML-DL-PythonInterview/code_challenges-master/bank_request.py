'''
You've been asked to program a bot for a popular bank that will automate the
management of incoming requests. There are three types of requests the bank
can receive:

transfer i j sum: request to transfer sum amount of money from the ith account
to the jth one;
deposit i sum: request to deposit sum amount of money in the ith account;
withdraw i sum: request to withdraw sum amount of money from the ith account.
Your bot should also be able to process invalid requests. There are two types
of invalid requests:

invalid account number in the requests;
deposit / withdrawal of a larger amount of money than is currently available.
For the given list of accounts and requests, return the state of accounts after
all requests have been processed, or an array of a single element
[-<request_id>] (please note the minus sign), where <request_id> is the 1-based
index of the first invalid request.

Example

For accounts = [10, 100, 20, 50, 30] and
requests = ["withdraw 2 10", "transfer 5 1 20", "deposit 5 20", "transfer 3 4 15"],
the output should be bankRequests(accounts, requests) = [30, 90, 5, 65, 30].

Here are the states of accounts after each request:

"withdraw 2 10": [10, 90, 20, 50, 30];
"transfer 5 1 20": [30, 90, 20, 50, 10];
"deposit 5 20": [30, 90, 20, 50, 30];
"transfer 3 4 15": [30, 90, 5, 65, 30], which is the answer.
For accounts = [20, 1000, 500, 40, 90] and
requests = ["deposit 3 400", "transfer 1 2 30", "withdraw 4 50"],
the output should be bankRequests(accounts, requests) = [-2].

After the first request, accounts becomes equal to [20, 1000, 900, 40, 90], but
the second one turns it into [-10, 1030, 900, 40, 90], which is invalid. Thus,
the second request is invalid, and the answer is [-2]. Note that the last
request is also invalid, but it shouldn't be included in the answer.

Input/Output

[execution time limit] 4 seconds (py3)

[input] array.integer accounts

Array of integers, where accounts[i] is the amount of money in the (i + 1)th account.

Guaranteed constraints:
0 < accounts.length ≤ 100,
0 ≤ accounts[i] ≤ 105.

[input] array.string requests

Array of requests in the order they should be processed. Each request is
guaranteed to be in the format described above.
It is guaranteed that for each i and j in a request 0 < i, j ≤ 1000.

Guaranteed constraints:
0 < requests.length ≤ 100,
for each i, j and sum in each element of request:

0 < i ≤ 1000,
0 < j ≤ 1000,
0 ≤ sum ≤ 105.
[output] array.integer

The accounts after executing all of the requests or array with a single
integer - the index of the first invalid request preceded by -.
'''


def bank_requests(accounts, requests):
    for i, request in enumerate(requests):
        request_split = request.split()
        action = request_split[0]
        accts = request_split[1:-1]
        amount = int(request_split[-1])

        if action == 'deposit':
            acct = int(accts[0])
            if acct > len(accounts):
                return [-i - 1]
            else:
                accounts[acct - 1] += amount

        elif action == 'withdraw':
            acct = int(accts[0])
            if acct > len(accounts) or (accounts[acct - 1] - amount) < 0:
                return [-i - 1]
            else:
                accounts[acct - 1] -= amount

        elif action == 'transfer':
            acct1 = int(accts[0])
            acct2 = int(accts[1])
            if acct1 > len(accounts) or acct2 > len(accounts) or (accounts[acct1 - 1] - amount) < 0:
                return [-i - 1]
            else:
                accounts[acct1 - 1] -= amount
                accounts[acct2 - 1] += amount

    return accounts
