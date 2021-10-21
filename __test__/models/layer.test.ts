import { Layer } from "../../src/models/layer"

describe("Testando o layer", () => {

    it("Testando uma camada com 3 neurônios e 3 entradas cada um", () => {

        const layer = new Layer(3, 3)

        expect(layer.weights.length).toEqual(3)
        expect(layer.weights[0].length).toEqual(3)

    })

})