function prior_model = get_prior_model(prior_name)

    switch prior_name
        case 'EM'
            load('data/4p7em_prior_k20.mat');
        case 'Sketching'
            load('data/sketching_5-3.mat');
    end
    prior_model.GS = GS;
    prior_model.name = prior_name;
    prior_model.t = t;
end
