"""RandomOrg

Random.org is a website that generates True Random Bits for RNG. Using their API, you can send requests
to receive random numbers or sequences back for use on different programs.
"""

import requests
import json


class RandomOrg(object):
    """ Random.org RNG API

    This class is used to connect to a Random.org Account for access to their API for generating random
    integers and decimals for various uses.
    """

    def __init__(self, api_key):
        """Constructor

        Args:
            api_key (str): The API key used by Random.org. Must be obtain through Random.org Account.
        """
        self._api_key = api_key

    @staticmethod
    def __call__(method, params, identifier=42):
        """ Generalizes the request done to Random.org for a given method and parameters given.

        Args:
            method (str): A method defined by Random.org that is available for use. (i.e. generateIntegers)
            params (dict): A dictionary containing things such as the API key. Go to Random.org for an example.
            identifier (int): An integer used to confirm that the request received is the one sent.

        Returns:
            (dict) A JSON response from Random.org converted into a Python dictionary.
        """
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": identifier
        }

        response = requests.post("https://api.random.org/json-rpc/2/invoke",
                                 data=json.dumps(request),
                                 headers={'content-type': 'application/json'},
                                 timeout=120.0)

        if 'error' in response.json():
            raise ConnectionError("Error received from Random.org!" +
                                  "\n\tCode: " + str(response.json()['error']['code']) +
                                  "\n\tMessage: " + response.json()['error']['message'])

        return response.json()

    def set_api_key(self, api_key):
        """ Sets the API key to a different value than what was initialized

        Args:
            api_key (str): The API key used by Random.org. Must be obtain through Random.org Account.
        """
        self._api_key = api_key

    def randint(self, low, high, n=1, replacement=True):
        """ Obtains n random integers within the user-defined range (inclusive).

        Args:
            low (int): The lowest integer value to receive.
            high (int): The highest integer value to receive.
            n (int): The number of integer values to receives. (Default: 1)
            replacement (bool): If true, the numbers may not be unique (like rolling two 6s from two dice rolls).
                If false, will make all values unique. (Default: true)

        Returns:
            (list) A list of integers from the received response.
        """
        params = {
            "apiKey": self._api_key,
            "n": n,
            "min": low,
            "max": high,
            "replacement": replacement
        }

        response = self.__call__("generateIntegers", params)

        return response['result']['random']['data']

    def random(self, n=1, decimal_places=8, replacement=True):
        """Obtains n random decimals from a uniform distribution [0, 1].

        Args:
            n (int): The number of integer values to receives. (Default: 1)
            decimal_places (int): The number of decimal places willing to have in the random decimal. (Default: 8)
            replacement (bool): If true, the numbers may not be unique (like rolling two 6s from two dice rolls).
                If false, will make all values unique.

        Returns:
            (list) A list of integers from the received response.
        """
        params = {
            "apiKey": self._api_key,
            "n": n,
            "decimalPlaces": decimal_places,
            "replacement": replacement
        }

        response = self.__call__("generateDecimalFractions", params)

        return response['result']['random']['data']

    def gauss(self, mu=0., sigma=1., n=1, significant_digits=8):
        """Obtains n decimals from a user-defined gaussian distribution.

        Args:
            mu (float): The mean of the guassian distribution. (Default: 0)
            sigma (float): The standard deviation of the guassian distribution. (Default: 1)
            n (int): The number of integer values to receives. (Default: 1)
            significant_digits (int): The number of decimal places willing to have in the random decimal. (Default: 8)

        Returns:

        """
        params = {
            "apiKey": self._api_key,
            "n": n,
            "mean": mu,
            "standardDeviation": sigma,
            "significantDigits": significant_digits
        }

        response = self.__call__("generateGaussian", params)

        return response['result']['random']['data']

    def choice(self, seq):
        """Returns a random object from the list given by the user.

        Args:
            seq (list): A list of objects to choose from.

        Returns:
            (Object) An object from the list given by the user.
        """
        if len(seq) == 0:
            raise IndexError("The length of the sequence is less than 1.")
        pick = self.randint(0, len(seq))
        return seq[pick]
