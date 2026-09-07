package com.sih26168.drnav

import android.annotation.SuppressLint
import android.content.Context
import android.location.GnssMeasurementsEvent
import android.location.GnssStatus
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.location.OnNmeaMessageListener
import android.os.Build
import android.os.Handler

/** A snapshot of sky quality, refreshed on every status callback. */
data class SkyState(
    val satellitesInFix: Int = 0,
    val satellitesVisible: Int = 0,
    val navicInFix: Int = 0,
    val navicVisible: Int = 0,
    val meanCn0: Double = 0.0,
    val hdop: Double = Double.NaN,
)

/**
 * Location fixes, satellite status, NMEA-derived HDOP, and raw GNSS measurements.
 *
 * The raw measurements callback is the reason this project is on Android at all.
 * It exposes per-satellite pseudorange rate, carrier frequency, C/N0 and
 * constellation identity, which is what makes NavIC visible as NavIC rather
 * than as an anonymous contribution to a fused blue dot, and what a multipath
 * classifier will eventually be trained on.
 *
 * HDOP is not exposed by the Android location API at all. It *is* present in
 * the NMEA `$__GSA` sentence, which the platform will stream to a listener, so
 * that is where we read it from. Without it the GNSS health classifier loses
 * its best single indicator of bad satellite geometry.
 */
class GnssCollector(
    context: Context,
    private val onFix: (Location) -> Unit,
    private val onSky: (SkyState) -> Unit,
    private val onRaw: (GnssMeasurementsEvent) -> Unit,
) {
    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
    private var sky = SkyState()

    private val locationListener = LocationListener { loc -> onFix(loc) }

    private val statusCallback = object : GnssStatus.Callback() {
        override fun onSatelliteStatusChanged(status: GnssStatus) {
            var inFix = 0; var navicFix = 0; var navicVis = 0
            var cn0Sum = 0.0; var cn0N = 0
            for (i in 0 until status.satelliteCount) {
                val isNavIC = Constellations.isNavIC(status.getConstellationType(i))
                if (isNavIC) navicVis++
                if (status.usedInFix(i)) {
                    inFix++
                    if (isNavIC) navicFix++
                }
                cn0Sum += status.getCn0DbHz(i).toDouble(); cn0N++
            }
            sky = sky.copy(
                satellitesInFix = inFix,
                satellitesVisible = status.satelliteCount,
                navicInFix = navicFix,
                navicVisible = navicVis,
                meanCn0 = if (cn0N > 0) cn0Sum / cn0N else 0.0,
            )
            onSky(sky)
        }
    }

    private val measurementsCallback = object : GnssMeasurementsEvent.Callback() {
        override fun onGnssMeasurementsReceived(event: GnssMeasurementsEvent) = onRaw(event)
    }

    /** `$GPGSA` / `$GNGSA` field 16 is HDOP. */
    private val nmeaListener = OnNmeaMessageListener { message, _ ->
        if (message.length > 6 && message.regionMatches(3, "GSA", 0, 3)) {
            val parts = message.split(',')
            if (parts.size > 16) {
                parts[16].substringBefore('*').toDoubleOrNull()?.let {
                    sky = sky.copy(hdop = it)
                    onSky(sky)
                }
            }
        }
    }

    val hasRawMeasurementSupport: Boolean
        get() = Build.VERSION.SDK_INT >= Build.VERSION_CODES.N

    @SuppressLint("MissingPermission")   // caller checks ACCESS_FINE_LOCATION
    fun start(handler: Handler) {
        lm.requestLocationUpdates(
            LocationManager.GPS_PROVIDER, 1000L, 0f, locationListener, handler.looper)
        lm.registerGnssStatusCallback(statusCallback, handler)
        lm.addNmeaListener(nmeaListener, handler)
        if (hasRawMeasurementSupport) {
            lm.registerGnssMeasurementsCallback(measurementsCallback, handler)
        }
    }

    fun stop() {
        runCatching { lm.removeUpdates(locationListener) }
        runCatching { lm.unregisterGnssStatusCallback(statusCallback) }
        runCatching { lm.removeNmeaListener(nmeaListener) }
        runCatching { lm.unregisterGnssMeasurementsCallback(measurementsCallback) }
    }
}
