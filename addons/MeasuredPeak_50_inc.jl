module MeasuredPeak

using IESopt
using JuMP
using Dates

function initialize!(model, config)
    @info "Addon detected"
    return true
end

function construct_variables!(model, config)
    @info "Constructing variables"

    # Monthly peak demand variables (in kW)
    @variable(model, monthly_peaks[1:12] >= 0)

    # Split average monthly peak into below and above 7.5 kW components
    @variable(model, avg_peak_below_7_5 >= 0)
    @variable(model, avg_peak_above_7_5 >= 0)

    return true
end

function construct_constraints!(model, config)
    @info "Constructing constraints"

    # Get time series data (15-min resolution)
    consumption = get_component(model, "grid_tariff").var.flow
    times = DateTime(2021):Minute(15):DateTime(2022)-Minute(15)
    months = month.(times)

    # Constraint: monthly_peak >= consumption for each 15-min interval
    for time in eachindex(times)
        @constraint(model, model[:monthly_peaks][months[time]] >= consumption[time])
    end

    # Constraint: average peak = sum(monthly_peaks) / 12
    @constraint(model, (sum(model[:monthly_peaks]) / 12) == model[:avg_peak_below_7_5] + model[:avg_peak_above_7_5])

    # Cap the lower-tier component at 7.5 kW
    @constraint(model, model[:avg_peak_below_7_5] <= 7.5)

    return true
end

function construct_objective!(model, config)
    @info "Constructing objective function"

    # Define rates
    rate_below = 30.31  # €/kW
    rate_above = 60.61  # €/kW

    # Objective: annual cost based on average monthly peak
    obj_expr = model[:avg_peak_below_7_5] * rate_below + model[:avg_peak_above_7_5] * rate_above

    # Add to model's total_cost objective
    push!(internal(model).model.objectives["total_cost"].terms, obj_expr)

    return true
end

end
