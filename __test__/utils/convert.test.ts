import { arrayDecToBin, binToDec, decToBin } from '../../src/utils/convert'

describe("Testando o convert", () => {

    it("Testando o método decimal para binário", () => {

        let bin = [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]
        let dec = 34.60

        expect(decToBin(dec)).toEqual(bin)

    })

    it("Testando o método binário para decimal", () => {

        let bin = [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0]

        let dec = 34.60

        expect(binToDec(bin)).toEqual(dec)

    })

    it("Testando o método para transformar um array decimal para um array binário", () => {

        let arrDec: number[][] = [
            [24.72],
            [24.36],
            [24.22]
        ]

        let arrBin: number[][] = [
            [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
        ]

        expect(arrayDecToBin(arrDec)).toEqual(arrBin)

    })

})