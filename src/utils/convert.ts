export function decToBin(value: number): number[] {

    return (value * 100).toString(2).split('').map(current => { return Number(current) })

}

export function binToDec(bin: number[]): number {

    return parseInt(bin.join(''), 2) / 100

}

export function arrayDecToBin(arrDec: number[][]): number[][] {

    let arrBin: number[][] = []

    arrDec.forEach(value => {

        arrBin.push(decToBin(value[0]))

    })

    return arrBin

}

