"""Tests for coordinator ordering and per-sensor metadata."""

from __future__ import annotations

import asyncio
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.components.sensor import SensorDeviceClass, SensorStateClass
from homeassistant.const import CONF_USERNAME, STATE_UNAVAILABLE, UnitOfEnergy

from custom_components.china_southern_power_grid_stat.const import (
    CONF_ELE_ACCOUNTS,
    DATA_KEY_LAST_UPDATE_DAY,
    DOMAIN,
    STATE_UPDATE_UNCHANGED,
    SUFFIX_BAL,
    SUFFIX_CURRENT_LADDER_REMAINING_KWH,
    SUFFIX_CURRENT_LADDER_TARIFF,
    SUFFIX_LAST_MONTH_KWH,
    SUFFIX_THIS_MONTH_COST,
    SUFFIX_THIS_MONTH_KWH,
    SUFFIX_THIS_YEAR_KWH,
    SUFFIX_YESTERDAY_KWH,
)
from custom_components.china_southern_power_grid_stat.sensor import (
    CSGCoordinator,
    CSGCostSensor,
    CSGEnergySensor,
)


def make_coordinator():
    """Return the minimal coordinator shape used by entity constructors."""
    return SimpleNamespace(context=None, data={}, _config_entry_id="entry-id")


@pytest.mark.parametrize(
    ("suffix", "state_class"),
    [
        (SUFFIX_THIS_MONTH_KWH, SensorStateClass.TOTAL_INCREASING),
        (SUFFIX_THIS_YEAR_KWH, SensorStateClass.TOTAL_INCREASING),
        (SUFFIX_YESTERDAY_KWH, None),
        (SUFFIX_LAST_MONTH_KWH, None),
    ],
)
def test_energy_metadata_matches_period_semantics(suffix, state_class):
    sensor = CSGEnergySensor(make_coordinator(), "account", suffix)

    assert sensor.device_class is SensorDeviceClass.ENERGY
    assert sensor.native_unit_of_measurement is UnitOfEnergy.KILO_WATT_HOUR
    assert sensor.state_class is state_class
    assert sensor.unique_id == f"{DOMAIN}.entry-id.account.{suffix}"


def test_remaining_energy_is_a_current_storage_measurement():
    sensor = CSGEnergySensor(
        make_coordinator(), "account", SUFFIX_CURRENT_LADDER_REMAINING_KWH
    )

    assert sensor.device_class is SensorDeviceClass.ENERGY_STORAGE
    assert sensor.state_class is SensorStateClass.MEASUREMENT


def test_cost_metadata_distinguishes_totals_balance_and_tariff():
    current_total = CSGCostSensor(
        make_coordinator(), "account", SUFFIX_THIS_MONTH_COST
    )
    balance = CSGCostSensor(make_coordinator(), "account", SUFFIX_BAL)
    tariff = CSGCostSensor(
        make_coordinator(), "account", SUFFIX_CURRENT_LADDER_TARIFF
    )

    assert current_total.device_class is SensorDeviceClass.MONETARY
    assert current_total.state_class is SensorStateClass.TOTAL_INCREASING
    assert balance.device_class is SensorDeviceClass.MONETARY
    assert balance.state_class is None
    assert tariff.device_class is None
    assert tariff.state_class is None
    assert tariff.native_unit_of_measurement == "CNY/kWh"


def test_device_identifier_is_scoped_to_config_entry():
    sensor = CSGEnergySensor(
        make_coordinator(), "account", SUFFIX_YESTERDAY_KWH
    )

    assert sensor.device_info["identifiers"] == {(DOMAIN, "entry-id:account")}


def test_unavailable_is_applied_before_state_is_written():
    coordinator = make_coordinator()
    coordinator.data = {
        "account": {SUFFIX_YESTERDAY_KWH: STATE_UNAVAILABLE}
    }
    sensor = CSGEnergySensor(coordinator, "account", SUFFIX_YESTERDAY_KWH)
    sensor._attr_available = True
    availability_when_written = []
    sensor.async_write_ha_state = Mock(
        side_effect=lambda: availability_when_written.append(sensor._attr_available)
    )

    sensor._handle_coordinator_update()

    assert availability_when_written == [False]


def test_successful_value_recovers_unavailable_sensor():
    coordinator = make_coordinator()
    coordinator.data = {"account": {SUFFIX_YESTERDAY_KWH: 12.34}}
    sensor = CSGEnergySensor(coordinator, "account", SUFFIX_YESTERDAY_KWH)
    sensor._attr_available = False
    sensor.async_write_ha_state = Mock()

    sensor._handle_coordinator_update()

    assert sensor._attr_available is True
    assert sensor.native_value == 12.34
    sensor.async_write_ha_state.assert_called_once_with()


def test_unchanged_value_preserves_sensor_availability_and_skips_write():
    coordinator = make_coordinator()
    coordinator.data = {
        "account": {SUFFIX_YESTERDAY_KWH: STATE_UPDATE_UNCHANGED}
    }
    sensor = CSGEnergySensor(coordinator, "account", SUFFIX_YESTERDAY_KWH)
    sensor._attr_available = False
    sensor.async_write_ha_state = Mock()

    sensor._handle_coordinator_update()

    assert sensor._attr_available is False
    sensor.async_write_ha_state.assert_not_called()


def test_month_merge_keeps_usage_when_cost_endpoint_is_unavailable():
    usage = [{"date": "2026-09-21", "kwh": 8.5}]

    by_day, kwh = CSGCoordinator.merge_by_day_data(
        by_day_from_cost=STATE_UNAVAILABLE,
        kwh_from_cost=STATE_UNAVAILABLE,
        by_day_from_usage=usage,
        kwh_from_usage=8.5,
    )

    assert by_day == usage
    assert kwh == 8.5


@pytest.mark.asyncio
async def test_this_month_failure_cannot_block_last_month_update():
    coordinator = object.__new__(CSGCoordinator)
    coordinator._if_update_last_month = False
    coordinator._gathered_data = {"account": {}}
    coordinator._async_update_bal_arr = AsyncMock()
    coordinator._async_update_yesterday_kwh = AsyncMock()
    coordinator._async_update_this_year_stats = AsyncMock()
    coordinator._async_update_last_year_stats = AsyncMock()
    coordinator._async_update_this_month_stats_and_ladder = AsyncMock(
        side_effect=RuntimeError("this-month failed")
    )
    coordinator._async_update_last_month_stats = AsyncMock()
    coordinator._update_latest_day = lambda _account: None
    account = SimpleNamespace(account_number="account")

    await asyncio.wait_for(coordinator._async_update_account_data(account), timeout=1)

    coordinator._async_update_last_month_stats.assert_awaited_once_with(account)
    assert coordinator._if_update_last_month is True


@pytest.mark.asyncio
async def test_update_data_deepcopies_mappingproxy_config():
    """Coordinator update must deep-copy the read-only mappingproxy entry data."""
    coordinator = object.__new__(CSGCoordinator)
    coordinator._config_entry_id = "entry-id"
    coordinator._config = MappingProxyType(
        {CONF_USERNAME: "user", CONF_ELE_ACCOUNTS: {}}
    )
    coordinator._config_entry = SimpleNamespace(data={}, options={})
    coordinator._gathered_data = {}
    coordinator._this_day = None
    coordinator._async_refresh_client = AsyncMock()

    class FakeConfigEntries:
        def async_get_entry(self, entry_id):
            return None

    coordinator.hass = SimpleNamespace(
        data={DOMAIN: {"entry-id": {}}},
        config_entries=FakeConfigEntries(),
    )

    result = await coordinator._async_update_data()

    assert result == {}
