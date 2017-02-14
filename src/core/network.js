export const NeuronType = {
    INPUT: "INPUT",
    HIDDEN: "HIDDEN",
    OUTPUT: "OUTPUT",
    BIAS: "BIAS"
}

export const SynapseType = {
    INPUT_TO_HIDDEN: "INPUT_TO_HIDDEN",
    HIDDEN_TO_HIDDEN: "HIDDEN_TO_HIDDEN",
    HIDDEN_TO_OUTPUT: "HIDDEN_TO_OUTPUT"
}

export class Neuron {
    constructor(neuronType) {
        this.type = neuronType;
        this.value = 0.0;
        this.synapses = [];
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivation(x) {
        return (1 - x) * x;
    }

    setValue(value) {
        this.value = value;
    }

    getValue() {
        return this.value;
    }

    computeValue(type) {
        this.value = this.sigmoid(this.synapses.filter(s => s.type === type).reduce((prev, next) => {
            return prev + (next.getWeight() * next.getInputValue());
        }, 0));
    }

    computeDelta(desired) {
        this.delta = (desired - this.value) * this.sigmoidDerivation(this.value);
    }

    setDelta(deltaValue) {
        this.delta = deltaValue;
    }

    getDelta() {
        return this.delta;
    }

    computeError(desired) {
        this.error = Math.pow((desired - this.getValue()), 2) / 1;
    }

    addSynapse(synapse) {
        this.synapses.push(synapse);
    }

    addSynapses(synapses) {
        this.synapses = synapses;
    }
}

export class Synapse {
    constructor(type, source, destiny) {
        this.weight = this.generateWeight();
        this.type = type;
        this.prevDelta = 0.0;
        this.addSourceNeuron(source);
        this.addDestinyNeuron(destiny);
    }

    getWeight() {
        return this.weight;
    }

    setGradient(gradValue) {
        this.gradient = gradValue;
    }

    getGradient() {
        return this.gradient;
    }

    adjustWeight() {
        let newWeight = (0.1 * this.gradient) + (this.prevDelta * 0.9);
        this.prevDelta = newWeight;
        this.weight += newWeight;
    }

    generateWeight() {
        return +(Math.random() * (0.9 - 0) + 0).toFixed(4);
    }

    getInputValue() {
        return this.source.getValue();
    }

    addSourceNeuron(neuron) {
        this.source = neuron;
    }

    addDestinyNeuron(neuron) {
        this.destiny = neuron;
    }
}

export class NN {

    constructor(config) {
        this.config = config;

        this.layers = {};
        this.synapses = {};

        this.layers[NeuronType.INPUT] = this.createLayer(NeuronType.INPUT, this.config.layers.input.size);
        this.layers[NeuronType.HIDDEN] = this.createLayer(NeuronType.HIDDEN, this.config.layers.hidden.size);
        this.layers[NeuronType.OUTPUT] = this.createLayer(NeuronType.OUTPUT, this.config.layers.output.size);

        this.createSynapses();
    }

    createLayer(neuronType) {
        const layerConfiguration = this.config.layers[neuronType.toLowerCase()];
        return this.createNeurons(neuronType, layerConfiguration.size);
    }

    createNeurons(neuronType, neuronSize) {
        let container = [];
        for (let i = 0; i < neuronSize; i++)
            container.push(new Neuron(neuronType));
        return container;
    }

    createSynapses() {
        const layerConfig = this.config.layers;
        this.synapses[SynapseType.INPUT_TO_HIDDEN] = [];
        this.synapses[SynapseType.HIDDEN_TO_OUTPUT] = [];
        let sourceNeuron, destinyNeuron;


        for (let h = 0; h < layerConfig.hidden.size; h++) {
            for (let i = 0; i < layerConfig.input.size; i++) {
                sourceNeuron = this.layers[NeuronType.INPUT][i];
                destinyNeuron = this.layers[NeuronType.HIDDEN][h];
                this.makeReference(SynapseType.INPUT_TO_HIDDEN, sourceNeuron, destinyNeuron);
            }
            for (let o = 0; o < layerConfig.output.size; o++) {
                sourceNeuron = this.layers[NeuronType.HIDDEN][h];
                destinyNeuron = this.layers[NeuronType.OUTPUT][o];
                this.makeReference(SynapseType.HIDDEN_TO_OUTPUT, sourceNeuron, destinyNeuron);
            }
        }


        this.train([
             [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1],// 0
             [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], //1
             [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1], //2
             [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], //3
             [1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1], // 4
             [1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],// 5
             [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1], //6
             [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],// 7
             [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],// 8
             [1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]// 9*/
          /*  [0, 1],
            [1, 0],
            [1, 1],
            [0, 0]*/
        ], [
        /*    [0],
            [0],
            [1],
            [0]*/
             [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ], () => {
            this.run([1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
            this.run([0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
            this.run([1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]);
            console.log();
            console.log();
            this.run([1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
            this.run([1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1]);
            console.log();
            console.log();
        });

    }

    train(trainInputSet, trainAnswers, fn) {
        console.time('TRAIN_TIME');
        let epoch = 0;
        while (epoch++ <= 400000) {
            for (let trainInputIndex = 0; trainInputIndex < trainInputSet.length; trainInputIndex++) {
                let trainInputs = trainInputSet[trainInputIndex];
                this.setTrainInput(trainInputs);

                this.layers[NeuronType.HIDDEN].forEach(n => n.computeValue(SynapseType.INPUT_TO_HIDDEN));
                this.layers[NeuronType.OUTPUT].forEach((n, index) => {
                    n.computeValue(SynapseType.HIDDEN_TO_OUTPUT);
                    n.computeError(trainAnswers[trainInputIndex][index]);
                    n.computeDelta(trainAnswers[trainInputIndex][index]);
                });

                this.synapses[SynapseType.HIDDEN_TO_OUTPUT].forEach(s => {
                    s.source.setDelta(s.source.sigmoidDerivation(s.source.getValue()) * (s.getWeight() * s.destiny.getDelta()));
                    s.setGradient(s.source.getValue() * s.destiny.getDelta());
                    s.adjustWeight();
                });

                this.synapses[SynapseType.INPUT_TO_HIDDEN].forEach(s => {
                    s.setGradient(s.source.getValue() * s.destiny.getDelta());
                    s.adjustWeight();
                });
            }

            if (epoch === 400000) {
                console.timeEnd('TRAIN_TIME');
                fn();
            }
        }
    }

    run(data) {
        console.time('RUN_TIME');
        this.setTrainInput(data);
        this.layers[NeuronType.HIDDEN].forEach(n => n.computeValue(SynapseType.INPUT_TO_HIDDEN));
        this.layers[NeuronType.OUTPUT].forEach((n, index) => {
            n.computeValue(SynapseType.HIDDEN_TO_OUTPUT);
            console.log(`${index} = ${n.getValue()}`);
        });
        console.timeEnd('RUN_TIME');
    }

    setTrainInput(trainInputs) {
        this.layers[NeuronType.INPUT].forEach((n, index) => n.setValue(trainInputs[index]));
    }

    makeReference(type, sourceNeuron, destinyNeuron) {
        let currentSynapse = new Synapse(type, sourceNeuron, destinyNeuron)
        this.synapses[type].push(currentSynapse);
        let synapseRef = this.synapses[type][this.synapses[type].length - 1];

        sourceNeuron.addSynapse(synapseRef);
        destinyNeuron.addSynapse(synapseRef);
    }

    /*train(input) {
     let ind = 0;
     this.layers[NeuronType.INPUT].forEach(neuron => neuron.setValue(input[ind++]));
     }*/

}

