import unittest

from esercizio import somma


class TestSomma(unittest.TestCase):
    def test_somma_numeri_positivi(self):
        self.assertEqual(somma(2, 3), 5)

    def test_somma_con_zero(self):
        self.assertEqual(somma(0, 0), 0)
        self.assertEqual(somma(0, 7), 7)
        self.assertEqual(somma(9, 0), 9)

    def test_somma_numeri_negativi(self):
        self.assertEqual(somma(-2, -3), -5)
        self.assertEqual(somma(-5, 2), -3)
        self.assertEqual(somma(5, -2), 3)

    def test_somma_numeri_grandi(self):
        self.assertEqual(somma(10_000_000, 5_000_000), 15_000_000)


if __name__ == "__main__":
    unittest.main()


