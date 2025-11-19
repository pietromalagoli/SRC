function NN = nn(layers,params,options)
    %layers: list of sizes of the NN layers (including input)
    %params: activation, etc
    %options as in trainnet
    which params.activation
        case "relu"
            activation = reluLayer()
        case "Heaviside"
            activation = HeavisideLayer()

    %net=tabella con colonne layer e stati dei neuroni
    numel()
    