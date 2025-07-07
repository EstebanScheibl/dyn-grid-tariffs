module MeasuredPeak

using IESopt
using JuMP
using Dates

function initialize!(model,config)
    @info "Addon detected"
    return true
end

function construct_variables!(model,config)
    @info "Constructing variables"
    # Add your variable construction logic here
    @variable(model, monthly_peaks[1:12])
    return true
end

function construct_constraints!(model,config)
    @info "Constructing constraints"
    # Add your constraint construction logic here
    consumption = get_component(model,"grid_tariff").var.flow
    times = DateTime(2021):Minute(15):DateTime(2022)-Minute(15)
    months = month.(times)
    @show months[end-196:end]
    for time in eachindex(times)
        @constraint(model, model[:monthly_peaks][months[time]] >= consumption[time])
    end
    @info "Monthly Peaks Variables Initialized:", model[:monthly_peaks] #
    return true
end

function construct_objective!(model,config)
    @info "Constructing objectives"
    # Add your objective construction logic here
    push!(internal(model).model.objectives["total_cost"].terms, (sum(model[:monthly_peaks])/12)*35.47)
    return true
end

end