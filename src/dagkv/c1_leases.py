"""Dependence-aware shared-lease aggregation for M3/C1."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from dagkv.domain import (
    BlockKey,
    IdentityError,
    ResidencyState,
    WorkflowKey,
    require_sha256,
    require_text,
)

PROBABILITY_SCALE = 1_000_000


class ForecastSource(StrEnum):
    """Information boundary used to construct a lease forecast."""

    PREDICTED = "predicted"
    ORACLE = "oracle"


class LeasePriorityMode(StrEnum):
    """Independently selectable aggregation used by a downstream controller."""

    C1_NOMINAL = "c1_nominal"
    C1_ROBUST_LOWER = "c1_robust_lower"
    PBKV_STYLE_ADDITIVE = "pbkv_style_additive"
    INDEPENDENT_MARGINAL_UNION = "independent_marginal_union"


@dataclass(frozen=True, slots=True)
class ReuseClaim:
    """One predicted owner access to a shared physical block."""

    claim_id: str
    binding_id: str
    workflow: WorkflowKey
    node_id: str
    reuse_epoch_id: str
    access_ns: int

    def __post_init__(self) -> None:
        require_text("claim_id", self.claim_id)
        require_text("claim binding_id", self.binding_id)
        require_text("claim node_id", self.node_id)
        require_text("reuse_epoch_id", self.reuse_epoch_id)
        _require_non_negative_int("claim access_ns", self.access_ns)


@dataclass(frozen=True, slots=True)
class JointOutcome:
    """One mutually exclusive outcome in a dependence group."""

    outcome_id: str
    mass_ppm: int
    claims: tuple[ReuseClaim, ...] = ()

    def __post_init__(self) -> None:
        require_text("outcome_id", self.outcome_id)
        _require_probability("outcome mass_ppm", self.mass_ppm, positive=True)
        if not isinstance(self.claims, tuple):
            raise IdentityError("outcome claims must be a tuple")
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise IdentityError("outcome claim IDs must be unique")
        epoch_times: dict[str, int] = {}
        for claim in self.claims:
            prior = epoch_times.setdefault(claim.reuse_epoch_id, claim.access_ns)
            if prior != claim.access_ns:
                raise IdentityError(
                    "coalesced reuse epoch has conflicting access timestamps"
                )


@dataclass(frozen=True, slots=True)
class DependenceGroup:
    """A complete joint distribution; separate groups declare independence."""

    group_id: str
    outcomes: tuple[JointOutcome, ...]
    total_variation_radius_ppm: int = 0

    def __post_init__(self) -> None:
        require_text("dependence group_id", self.group_id)
        if not isinstance(self.outcomes, tuple) or not self.outcomes:
            raise IdentityError("dependence group requires a non-empty outcome tuple")
        _require_probability(
            "total_variation_radius_ppm",
            self.total_variation_radius_ppm,
            positive=False,
        )
        outcome_ids = [outcome.outcome_id for outcome in self.outcomes]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise IdentityError("dependence-group outcome IDs must be unique")
        if sum(outcome.mass_ppm for outcome in self.outcomes) != PROBABILITY_SCALE:
            raise IdentityError(
                "dependence-group outcome masses must sum to 1,000,000 ppm"
            )

        claim_identities: dict[str, ReuseClaim] = {}
        epoch_times: dict[str, int] = {}
        for outcome in self.outcomes:
            for claim in outcome.claims:
                prior_claim = claim_identities.setdefault(claim.claim_id, claim)
                if prior_claim != claim:
                    raise IdentityError(
                        "claim identity changes across dependence outcomes"
                    )
                prior_time = epoch_times.setdefault(
                    claim.reuse_epoch_id,
                    claim.access_ns,
                )
                if prior_time != claim.access_ns:
                    raise IdentityError(
                        "reuse epoch changes timestamp across dependence outcomes"
                    )


@dataclass(frozen=True, slots=True)
class SharedLeaseForecast:
    """A block forecast bound to one exact canonical runtime snapshot."""

    forecast_id: str
    block_key: BlockKey
    runtime_event_count: int
    generated_ns: int
    horizon_ns: int
    source: ForecastSource
    predictor_digest: str
    dependence_digest: str
    independence_basis: str
    groups: tuple[DependenceGroup, ...]

    def __post_init__(self) -> None:
        require_text("forecast_id", self.forecast_id)
        _require_non_negative_int("runtime_event_count", self.runtime_event_count)
        _require_non_negative_int("forecast generated_ns", self.generated_ns)
        _require_non_negative_int("forecast horizon_ns", self.horizon_ns)
        if self.horizon_ns <= self.generated_ns:
            raise IdentityError("forecast horizon must follow generation")
        if not isinstance(self.source, ForecastSource):
            raise IdentityError("forecast source must be a ForecastSource")
        require_sha256("predictor_digest", self.predictor_digest)
        require_sha256("dependence_digest", self.dependence_digest)
        require_text("independence_basis", self.independence_basis)
        if not isinstance(self.groups, tuple) or not self.groups:
            raise IdentityError("shared lease forecast requires dependence groups")
        group_ids = [group.group_id for group in self.groups]
        if len(group_ids) != len(set(group_ids)):
            raise IdentityError("forecast dependence-group IDs must be unique")

        claim_groups: dict[str, str] = {}
        epoch_groups: dict[str, str] = {}
        for group in self.groups:
            for outcome in group.outcomes:
                for claim in outcome.claims:
                    if not self.generated_ns < claim.access_ns <= self.horizon_ns:
                        raise IdentityError(
                            "claim access must fall after generation and within horizon"
                        )
                    prior_group = claim_groups.setdefault(
                        claim.claim_id, group.group_id
                    )
                    if prior_group != group.group_id:
                        raise IdentityError(
                            "one claim cannot span declared-independent groups"
                        )
                    prior_epoch_group = epoch_groups.setdefault(
                        claim.reuse_epoch_id,
                        group.group_id,
                    )
                    if prior_epoch_group != group.group_id:
                        raise IdentityError(
                            "one reuse epoch cannot span declared-independent groups"
                        )


@dataclass(frozen=True, slots=True)
class LeaseOwnerSnapshot:
    """Detached policy projection of one active retention binding."""

    binding_id: str
    workflow: WorkflowKey
    created_ns: int
    eligible_node_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_text("lease owner binding_id", self.binding_id)
        _require_non_negative_int("lease owner created_ns", self.created_ns)
        if not isinstance(self.eligible_node_ids, tuple):
            raise IdentityError("eligible_node_ids must be a tuple")
        for node_id in self.eligible_node_ids:
            require_text("eligible node_id", node_id)
        if len(self.eligible_node_ids) != len(set(self.eligible_node_ids)):
            raise IdentityError("eligible node IDs must be unique")


@dataclass(frozen=True, slots=True)
class SharedLeasePolicySnapshot:
    """Immutable policy input projected by the sole-writer runtime."""

    block_key: BlockKey
    runtime_event_count: int
    location_version: int
    residency: ResidencyState
    owners: tuple[LeaseOwnerSnapshot, ...]

    def __post_init__(self) -> None:
        _require_non_negative_int(
            "snapshot runtime_event_count", self.runtime_event_count
        )
        _require_non_negative_int("snapshot location_version", self.location_version)
        if not isinstance(self.residency, ResidencyState):
            raise IdentityError("snapshot residency must be a ResidencyState")
        if not isinstance(self.owners, tuple):
            raise IdentityError("snapshot owners must be a tuple")
        binding_ids = [owner.binding_id for owner in self.owners]
        if len(binding_ids) != len(set(binding_ids)):
            raise IdentityError("snapshot owner binding IDs must be unique")


@dataclass(frozen=True, slots=True)
class MetricInterval:
    """Exact nominal value and sound total-variation uncertainty bounds."""

    lower: Fraction
    nominal: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.nominal or self.nominal > self.upper:
            raise IdentityError("metric interval does not contain its nominal value")


@dataclass(frozen=True, slots=True)
class SharedLeasePoint:
    """Dependence-correct lease statistics through one deadline."""

    deadline_ns: int
    first_physical_readmission: MetricInterval
    expected_unique_reuse_epochs: MetricInterval
    expected_repeated_reuse_epochs: MetricInterval
    additive_claim_score: Fraction
    independent_marginal_union_score: Fraction

    def priority(self, mode: LeasePriorityMode) -> Fraction:
        """Select one independently ablatable ranking signal."""

        if mode == LeasePriorityMode.C1_NOMINAL:
            return self.first_physical_readmission.nominal
        if mode == LeasePriorityMode.C1_ROBUST_LOWER:
            return self.first_physical_readmission.lower
        if mode == LeasePriorityMode.PBKV_STYLE_ADDITIVE:
            return self.additive_claim_score
        if mode == LeasePriorityMode.INDEPENDENT_MARGINAL_UNION:
            return self.independent_marginal_union_score
        raise IdentityError(f"unsupported lease priority mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class SharedLeaseProfile:
    """A deadline curve for one snapshot-bound shared-block forecast."""

    forecast_id: str
    block_key: BlockKey
    runtime_event_count: int
    points: tuple[SharedLeasePoint, ...]

    def at_deadline(self, deadline_ns: int) -> SharedLeasePoint:
        """Return the exact preregistered deadline point."""

        for point in self.points:
            if point.deadline_ns == deadline_ns:
                return point
        raise IdentityError(f"deadline is absent from lease profile: {deadline_ns}")


def aggregate_shared_lease(
    snapshot: SharedLeasePolicySnapshot,
    forecast: SharedLeaseForecast,
    *,
    allow_oracle: bool = False,
) -> SharedLeaseProfile:
    """Aggregate joint outcomes without inventing branch independence."""

    _validate_forecast_scope(snapshot, forecast, allow_oracle=allow_oracle)
    deadlines = {
        claim.access_ns
        for group in forecast.groups
        for outcome in group.outcomes
        for claim in outcome.claims
    }
    deadlines.add(forecast.horizon_ns)
    points = tuple(
        _aggregate_deadline(forecast.groups, deadline_ns)
        for deadline_ns in sorted(deadlines)
    )
    return SharedLeaseProfile(
        forecast_id=forecast.forecast_id,
        block_key=forecast.block_key,
        runtime_event_count=forecast.runtime_event_count,
        points=points,
    )


def _validate_forecast_scope(
    snapshot: SharedLeasePolicySnapshot,
    forecast: SharedLeaseForecast,
    *,
    allow_oracle: bool,
) -> None:
    if snapshot.block_key != forecast.block_key:
        raise IdentityError("forecast block differs from policy snapshot")
    if snapshot.runtime_event_count != forecast.runtime_event_count:
        raise IdentityError("forecast is stale for the canonical runtime snapshot")
    if forecast.source == ForecastSource.ORACLE and not allow_oracle:
        raise IdentityError("oracle forecast is excluded from online policy scoring")

    owners = {owner.binding_id: owner for owner in snapshot.owners}
    for group in forecast.groups:
        for outcome in group.outcomes:
            for claim in outcome.claims:
                owner = owners.get(claim.binding_id)
                if owner is None:
                    raise IdentityError(
                        f"forecast claim lacks an active retention owner: "
                        f"{claim.binding_id}"
                    )
                if owner.workflow != claim.workflow:
                    raise IdentityError("forecast claim crosses workflow ownership")
                if claim.node_id not in owner.eligible_node_ids:
                    raise IdentityError("forecast claim targets an ineligible DAG node")
                if forecast.generated_ns < owner.created_ns:
                    raise IdentityError("forecast predates its retention owner")


def _aggregate_deadline(
    groups: tuple[DependenceGroup, ...],
    deadline_ns: int,
) -> SharedLeasePoint:
    zero_intervals: list[MetricInterval] = []
    epoch_intervals: list[MetricInterval] = []
    marginal_claims: list[Fraction] = []

    for group in groups:
        epoch_counts = tuple(
            len(
                {
                    claim.reuse_epoch_id
                    for claim in outcome.claims
                    if claim.access_ns <= deadline_ns
                }
            )
            for outcome in group.outcomes
        )
        zero_values = tuple(int(count == 0) for count in epoch_counts)
        zero_intervals.append(_metric_interval(group, zero_values))
        epoch_intervals.append(_metric_interval(group, epoch_counts))
        marginal_claims.extend(_claim_marginals(group, deadline_ns))

    zero_nominal = _fraction_product(interval.nominal for interval in zero_intervals)
    zero_lower = _fraction_product(interval.lower for interval in zero_intervals)
    zero_upper = _fraction_product(interval.upper for interval in zero_intervals)
    first = MetricInterval(
        lower=1 - zero_upper,
        nominal=1 - zero_nominal,
        upper=1 - zero_lower,
    )

    unique_epochs = MetricInterval(
        lower=sum((interval.lower for interval in epoch_intervals), Fraction()),
        nominal=sum(
            (interval.nominal for interval in epoch_intervals),
            Fraction(),
        ),
        upper=sum((interval.upper for interval in epoch_intervals), Fraction()),
    )
    repeated = MetricInterval(
        lower=max(Fraction(), unique_epochs.lower - first.upper),
        nominal=unique_epochs.nominal - first.nominal,
        upper=max(Fraction(), unique_epochs.upper - first.lower),
    )
    independent_union = 1 - _fraction_product(
        1 - marginal for marginal in marginal_claims
    )
    return SharedLeasePoint(
        deadline_ns=deadline_ns,
        first_physical_readmission=first,
        expected_unique_reuse_epochs=unique_epochs,
        expected_repeated_reuse_epochs=repeated,
        additive_claim_score=sum(marginal_claims, Fraction()),
        independent_marginal_union_score=independent_union,
    )


def _claim_marginals(
    group: DependenceGroup,
    deadline_ns: int,
) -> tuple[Fraction, ...]:
    masses: dict[str, int] = {}
    for outcome in group.outcomes:
        for claim in outcome.claims:
            if claim.access_ns <= deadline_ns:
                masses[claim.claim_id] = (
                    masses.get(claim.claim_id, 0) + outcome.mass_ppm
                )
    return tuple(
        Fraction(mass, PROBABILITY_SCALE) for _, mass in sorted(masses.items())
    )


def _metric_interval(
    group: DependenceGroup,
    values: tuple[int, ...],
) -> MetricInterval:
    masses = tuple(outcome.mass_ppm for outcome in group.outcomes)
    nominal_numerator = sum(
        mass * value for mass, value in zip(masses, values, strict=True)
    )
    lower_numerator, upper_numerator = _tv_weighted_bounds(
        masses,
        values,
        group.total_variation_radius_ppm,
    )
    return MetricInterval(
        lower=Fraction(lower_numerator, PROBABILITY_SCALE),
        nominal=Fraction(nominal_numerator, PROBABILITY_SCALE),
        upper=Fraction(upper_numerator, PROBABILITY_SCALE),
    )


def _tv_weighted_bounds(
    masses: tuple[int, ...],
    values: tuple[int, ...],
    radius_ppm: int,
) -> tuple[int, int]:
    """Solve scalar expectation extrema in an exact TV-radius mass ball."""

    nominal = sum(mass * value for mass, value in zip(masses, values, strict=True))
    minimum = min(values)
    maximum = max(values)

    lower = nominal
    remaining = radius_ppm
    for mass, value in sorted(
        zip(masses, values, strict=True),
        key=lambda item: item[1],
        reverse=True,
    ):
        if remaining == 0 or value == minimum:
            break
        moved = min(mass, remaining)
        lower -= moved * (value - minimum)
        remaining -= moved

    upper = nominal
    remaining = radius_ppm
    for mass, value in sorted(
        zip(masses, values, strict=True),
        key=lambda item: item[1],
    ):
        if remaining == 0 or value == maximum:
            break
        moved = min(mass, remaining)
        upper += moved * (maximum - value)
        remaining -= moved
    return lower, upper


def _fraction_product(values: Iterable[Fraction]) -> Fraction:
    product = Fraction(1)
    for value in values:
        product *= value
    return product


def _require_probability(name: str, value: int, *, positive: bool) -> None:
    if type(value) is not int:
        raise IdentityError(f"{name} must be an integer")
    minimum = 1 if positive else 0
    if not minimum <= value <= PROBABILITY_SCALE:
        qualifier = "positive " if positive else ""
        raise IdentityError(f"{name} must be a {qualifier}PPM probability")


def _require_non_negative_int(name: str, value: int) -> None:
    if type(value) is not int or value < 0:
        raise IdentityError(f"{name} must be a non-negative integer")
