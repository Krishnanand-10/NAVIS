package com.sih26168.drnav

import android.location.GnssStatus

/**
 * Constellation identity, including NavIC.
 *
 * [GnssStatus.CONSTELLATION_IRNSS] only exists from API 29, so the value is
 * inlined here and compared numerically. That keeps the app installable on
 * Android 8 and 9 phones while still reporting NavIC correctly on the
 * Qualcomm and MediaTek chipsets that carry it.
 */
object Constellations {
    const val GPS = GnssStatus.CONSTELLATION_GPS          // 1
    const val SBAS = GnssStatus.CONSTELLATION_SBAS        // 2
    const val GLONASS = GnssStatus.CONSTELLATION_GLONASS  // 3
    const val QZSS = GnssStatus.CONSTELLATION_QZSS        // 4
    const val BEIDOU = GnssStatus.CONSTELLATION_BEIDOU    // 5
    const val GALILEO = GnssStatus.CONSTELLATION_GALILEO  // 6
    const val IRNSS = 7                                   // NavIC, API 29+

    fun name(type: Int): String = when (type) {
        GPS -> "GPS"
        SBAS -> "SBAS"
        GLONASS -> "GLONASS"
        QZSS -> "QZSS"
        BEIDOU -> "BeiDou"
        GALILEO -> "Galileo"
        IRNSS -> "NavIC"
        else -> "unknown"
    }

    fun isNavIC(type: Int) = type == IRNSS
}
